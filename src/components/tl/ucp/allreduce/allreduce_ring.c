#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_dt_reduce.h"
#include "components/ec/ucc_ec.h"

void ucc_tl_ucp_allreduce_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         trank     = task->subset.myrank;
    ucc_rank_t         tsize     = (ucc_rank_t)task->subset.map.ep_num;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  mem_type  = TASK_ARGS(task).dst.info.mem_type;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;

    /* Early return for zero-count or single-rank edge cases */
    if (data_size == 0 || tsize <= 1) {
        /* If not in-place, we need to copy sbuf to rbuf */
        if (!UCC_IS_INPLACE(TASK_ARGS(task)) && data_size > 0) {
            memcpy(rbuf, sbuf, data_size);
        }
        task->super.status = UCC_OK;
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_done", 0);
        return;
    }

    size_t             chunk_size, send_offset, recv_offset;
    size_t             send_chunk_size, recv_chunk_size;
    ucc_rank_t         sendto, recvfrom;
    void              *recv_buf, *send_buf, *reduce_buf;
    ucc_status_t       status;
    int                step;
    int                num_chunks = tsize;
    int                send_chunk, recv_chunk;
    enum {
        RING_PHASE_INIT,      /* Initialize step */
        RING_PHASE_SEND_RECV, /* Send/receive phase */
        RING_PHASE_REDUCE,    /* Reduction phase */
        RING_PHASE_COMPLETE   /* Step is complete, advance to next step */
    } phase;

    /* Divide data into chunks, ensuring chunk size is aligned to datatype size */
    chunk_size = ucc_div_round_up(data_size, num_chunks);
    chunk_size = ((chunk_size + dt_size - 1) / dt_size) * dt_size;

    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        sbuf = rbuf;
    }

    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);

    /* Single-phase Ring Algorithm (SRA):
     * - Each rank starts with its local chunk fully reduced
     * - In each step, ranks exchange chunks and combine them
     * - After tsize-1 steps, all ranks have the complete reduced result
     */

    /* On first entry, initialize step and phase */
    if (task->allreduce_ring.step == 0 && task->allreduce_ring.phase == RING_PHASE_INIT) {
        if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
            memcpy(rbuf, sbuf, data_size);
        }
        task->allreduce_ring.phase = RING_PHASE_SEND_RECV;
    }

    /* Process steps: 0 to tsize-2 (standard SRA uses tsize-1 steps) */
    while (task->allreduce_ring.step < tsize - 1) {
        step = task->allreduce_ring.step;
        phase = task->allreduce_ring.phase;

        /* Check if we have a pending reduction task and test for completion */
        if (phase == RING_PHASE_REDUCE && task->allreduce_ring.etask != NULL) {
            status = ucc_ee_executor_task_test(task->allreduce_ring.etask);

            if (status == UCC_INPROGRESS) {
                /* Return and try again later */
                return;
            }

            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "reduction task failed: %s", 
                         ucc_status_string(status));
                task->super.status = status;
                return;
            }

            ucc_ee_executor_task_finalize(task->allreduce_ring.etask);
            task->allreduce_ring.etask = NULL;
            task->allreduce_ring.phase = RING_PHASE_COMPLETE;
        }

        /* If we've completed the current step, advance to the next one */
        if (phase == RING_PHASE_COMPLETE) {
            task->allreduce_ring.step++;
            task->allreduce_ring.phase = RING_PHASE_SEND_RECV;

            if (task->allreduce_ring.step >= tsize - 1) {
                break;
            }

            step = task->allreduce_ring.step;
        }

        /* Send/receive phase */
        if (phase == RING_PHASE_SEND_RECV) {
            send_chunk = (trank - step + tsize) % tsize;
            recv_chunk = (trank - step - 1 + tsize) % tsize;

            /* Calculate send offset and chunk size */
            send_offset = send_chunk * chunk_size;
            if (send_offset >= data_size) {
                task->allreduce_ring.phase = RING_PHASE_COMPLETE;
                continue;
            }

            /* Calculate actual size of this chunk */
            send_chunk_size = data_size - send_offset;
            if (send_chunk_size > chunk_size) {
                send_chunk_size = chunk_size;
            }
            send_chunk_size = (send_chunk_size / dt_size) * dt_size;

            if (send_chunk_size == 0) {
                task->allreduce_ring.phase = RING_PHASE_COMPLETE;
                continue;
            }

            /* Calculate receive offset and chunk size */
            recv_offset = recv_chunk * chunk_size;
            if (recv_offset >= data_size) {
                task->allreduce_ring.phase = RING_PHASE_COMPLETE;
                continue;
            }

            recv_chunk_size = data_size - recv_offset;
            if (recv_chunk_size > chunk_size) {
                recv_chunk_size = chunk_size;
            }
            recv_chunk_size = (recv_chunk_size / dt_size) * dt_size;

            if (recv_chunk_size == 0) {
                task->allreduce_ring.phase = RING_PHASE_COMPLETE;
                continue;
            }

            /* Send and receive chunks */
            send_buf = PTR_OFFSET(rbuf, send_offset);
            recv_buf = PTR_OFFSET(task->allreduce_ring.scratch, 0);

            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(send_buf, send_chunk_size, mem_type, sendto, team, task),
                task, out);

            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(recv_buf, recv_chunk_size, mem_type, recvfrom, team, task),
                task, out);

            if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
                return;
            }

            /* Save chunk information for the reduction phase */
            task->allreduce_ring.phase = RING_PHASE_REDUCE;
            task->allreduce_ring.recv_offset = recv_offset;
            task->allreduce_ring.recv_size = recv_chunk_size;
        }

        if (phase == RING_PHASE_REDUCE) {
            recv_offset = task->allreduce_ring.recv_offset;
            recv_chunk_size = task->allreduce_ring.recv_size;

            recv_buf = PTR_OFFSET(task->allreduce_ring.scratch, 0);
            reduce_buf = PTR_OFFSET(rbuf, recv_offset);

            status = ucc_dt_reduce(reduce_buf, recv_buf, reduce_buf,
                                recv_chunk_size / dt_size,
                                dt, &TASK_ARGS(task), 0, 0,
                                task->allreduce_ring.executor,
                                &task->allreduce_ring.etask);

            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }

            if (task->allreduce_ring.etask != NULL) {
                return;
            }

            task->allreduce_ring.phase = RING_PHASE_COMPLETE;
        }
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t        *sbgp;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    size_t             chunk_size;
    ucc_status_t       status;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET) && team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    /* Calculate chunk size for a single chunk */
    chunk_size = ucc_div_round_up(data_size, task->subset.map.ep_num);
    chunk_size = ((chunk_size + dt_size - 1) / dt_size) * dt_size;

    /* Allocate scratch space for a single chunk */
    status = ucc_mc_alloc(&task->allreduce_ring.scratch_mc_header,
                          chunk_size, TASK_ARGS(task).dst.info.mem_type);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        return status;
    }
    task->allreduce_ring.scratch = task->allreduce_ring.scratch_mc_header->addr;

    task->allreduce_ring.step = 0;
    task->allreduce_ring.phase = 0;
    task->allreduce_ring.etask = NULL;

    ucc_ee_executor_params_t eparams = {0};
    eparams.mask = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
    eparams.ee_type = UCC_EE_CPU_THREAD;
    status = ucc_ee_executor_init(&eparams, &task->allreduce_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to initialize executor");
        ucc_mc_free(task->allreduce_ring.scratch_mc_header);
        return status;
    }

    task->super.post     = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress = ucc_tl_ucp_allreduce_ring_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_ring_finalize;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allreduce_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st, global_st = UCC_OK;

    if (task->allreduce_ring.etask != NULL) {
        ucc_ee_executor_task_finalize(task->allreduce_ring.etask);
        task->allreduce_ring.etask = NULL;
    }

    if (task->allreduce_ring.executor != NULL) {
        st = ucc_ee_executor_finalize(task->allreduce_ring.executor);
        if (ucc_unlikely(st != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task), "failed to finalize executor");
            global_st = st;
        }
        task->allreduce_ring.executor = NULL;
    }

    st = ucc_mc_free(task->allreduce_ring.scratch_mc_header);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to free scratch buffer");
        global_st = (global_st == UCC_OK) ? st : global_st;
    }

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
        global_st = (global_st == UCC_OK) ? st : global_st;
    }
    return global_st;
}
