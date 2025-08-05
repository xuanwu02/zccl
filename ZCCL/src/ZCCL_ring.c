#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"
#include "ZCCL_ring.h"

/**
 * Pacer Bandwidth Limiter Integration
 * Credit to Yuke Li @ UC Merced
*/

int write_flag = 1;
int write_send_flag = 1;
int print_flag = 10;

int burst_fraction = 10;
static size_t burst_bytes;// = BANDWIDTH_LIMIT / burst_fraction; // Size of burst in bytes

void add_seg_info(MPI_Request* req, SegInfo* info) {
    for (int i = 0; i < segMapCount; ++i) {
        if (segMap[i].key == req) {
            segMap[i].value = info;
            return;
        }
    }
    if (segMapCount < MAX_SEG_ENTRIES) {
        segMap[segMapCount].key = req;
        segMap[segMapCount].value = info;
        ++segMapCount;
    }
}

SegInfo* find_seg_info(MPI_Request* req) {
    for (int i = 0; i < segMapCount; ++i) {
        if (segMap[i].key == req) {
            return segMap[i].value;
        }
    }
    return NULL;
}

void remove_seg_info(MPI_Request* req) {
    for (int i = 0; i < segMapCount; ++i) {
        if (segMap[i].key == req) {
            if (segMap[i].value != NULL) {
                free(segMap[i].value->requests);
                free(segMap[i].value->allocated_requests);
                free(segMap[i].value);
            }
            segMap[i] = segMap[segMapCount - 1];
            --segMapCount;
            return;
        }
    }
}

static int requestMapCount = 0;

void add_request_info(MPI_Request* req, RequestInfo* info) {
    for (int i = 0; i < requestMapCount; ++i) {
        if (requestMap[i].key == req) {
            requestMap[i].value = info;
            return;
        }
    }
    if (requestMapCount < MAX_REQUEST_MAP_ENTRIES) {
        requestMap[requestMapCount].key = req;
        requestMap[requestMapCount].value = info;
        ++requestMapCount;
    }
}

RequestInfo* find_request_info(MPI_Request* req) {
    for (int i = 0; i < requestMapCount; ++i) {
        if (requestMap[i].key == req) {
            return requestMap[i].value;
        }
    }
    return NULL;
}

void remove_request_info(MPI_Request* req) {
    for (int i = 0; i < requestMapCount; ++i) {
        if (requestMap[i].key == req) {
            RequestInfo* info = requestMap[i].value;
            if (info) {
                free(info->requests);
                free(info->send_buffers);
                free(info->segment_sizes);
                free(info->seg_requests);
                free(info->segment_buffers);
                free(info);
            }
            requestMap[i] = requestMap[requestMapCount - 1];
            --requestMapCount;
            return;
        }
    }
}

int MPI_Init(int* argc, char*** argv) {
    int ret = PMPI_Init(argc, argv);
    if (ret != MPI_SUCCESS) {
        printf("MPI_Init failed\n");
        return ret;
    }
    // printf("MPI_Init success\n");

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    int shm_rank, shm_size;
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_size);
    // printf("creat a communicator on the same node\n");

    int alloc_size = (shm_rank == 0) ? sizeof(struct timespec) : 0;
    MPI_Win_allocate_shared(alloc_size, sizeof(char),
                            MPI_INFO_NULL, shm_comm, &shared_last_time, &last_time_win);
    if (shm_rank != 0) {
        int disp_unit;
        MPI_Aint size;
        MPI_Win_shared_query(last_time_win, 0, &size, &disp_unit, &shared_last_time);
    }
    // printf("Checkpoint 1\n");
    if (shm_rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, shared_last_time);
    }

    MPI_Win_fence(0, last_time_win);
    // printf("Checkpoint 2\n");

    char* env = getenv("MAX_RATE");
    if (env != NULL) {
        BANDWIDTH_LIMIT = (double)atoll(env);
    }
    // printf("MAX_RATE: %zu\n", BANDWIDTH_LIMIT);
    burst_bytes = BANDWIDTH_LIMIT / burst_fraction;

    return ret;
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
    int ret = PMPI_Init_thread(argc, argv, required, provided);
    if (ret != MPI_SUCCESS) {
        printf("MPI_Init_thread failed\n");
        return ret;
    }
    // printf("MPI_Init_thread success, provided thread level: %d\n", *provided);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    int shm_rank, shm_size;
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &shm_size);
    // printf("Created a communicator on the same node (rank %d of %d)\n", shm_rank, shm_size);

    int alloc_size = (shm_rank == 0) ? sizeof(struct timespec) : 0;
    MPI_Win_allocate_shared(alloc_size, sizeof(char),
                            MPI_INFO_NULL, shm_comm, &shared_last_time, &last_time_win);
    if (shm_rank != 0) {
        int disp_unit;
        MPI_Aint size;
        MPI_Win_shared_query(last_time_win, 0, &size, &disp_unit, &shared_last_time);
    }

    // printf("Checkpoint 1\n");

    if (shm_rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, shared_last_time);
    }

    MPI_Win_fence(0, last_time_win);
    // printf("Checkpoint 2\n");

    char* env = getenv("MAX_RATE");
    if (env != NULL) {
        BANDWIDTH_LIMIT = (double)atoll(env);
    }
    // printf("MAX_RATE: %zu\n", BANDWIDTH_LIMIT);

    burst_bytes = (size_t)(BANDWIDTH_LIMIT / burst_fraction);

    return ret;
}

int mpi_isend_seg(const void* buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    int type_size = 0, err = 0;
    PMPI_Type_size(datatype, &type_size);
    size_t msg_bytes = (size_t)count * (size_t)type_size;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double now_us = timespec_to_us(&now);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, last_time_win);
    double virtual_time = timespec_to_us(shared_last_time);
    MPI_Win_unlock(0, last_time_win);

    double credit = (now_us > virtual_time) ? (now_us - virtual_time) : 0.0;
    double computed_delay_us = ((double)msg_bytes / BANDWIDTH_LIMIT) * 1e6 + 1.0;
    double actual_delay_us = computed_delay_us - credit;
    if (actual_delay_us < 0.0) actual_delay_us = 0.0;

    double base_time = MAX(virtual_time, now_us);
    double new_virtual_time = base_time + actual_delay_us;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, last_time_win);
    us_to_timespec(new_virtual_time, shared_last_time);
    MPI_Win_unlock(0, last_time_win);

    double sleep_time_us = new_virtual_time - now_us + 1.0;
    if (sleep_time_us > 0)
        precise_sleep_us(sleep_time_us);

    if (sendEventCount < MAX_SEND_EVENTS) {
        sendEvents[sendEventCount].timestamp = new_virtual_time;
        sendEvents[sendEventCount].size = msg_bytes;
        sendEventCount++;
    }

    err = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    if (err != MPI_SUCCESS)
        return err;

    RequestInfo* info = (RequestInfo*)calloc(1, sizeof(RequestInfo));
    if (!info) return MPI_ERR_OTHER;

    info->type = SEND;
    info->pipeline = 0;
    info->buffer_size = (int)msg_bytes;

    add_request_info(request, info);
    return err;
}

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    int type_size = 0;
    int err = 0;

    PMPI_Type_size(datatype, &type_size);
    size_t msg_bytes = (size_t)count * (size_t)type_size;

    if (msg_bytes > burst_bytes) {
        int num_segments = (msg_bytes + burst_bytes - 1) / burst_bytes;

        SegInfo* info = (SegInfo*)calloc(1, sizeof(SegInfo));
        if (!info) return MPI_ERR_OTHER;

        info->type = SEND;
        info->segmented = 1;
        info->num_segments = num_segments;

        info->requests = (MPI_Request*)malloc(sizeof(MPI_Request) * num_segments);
        info->allocated_requests = (MPI_Request**)malloc(sizeof(MPI_Request*) * (num_segments - 1));
        if (!info->requests || !info->allocated_requests) return MPI_ERR_OTHER;

        for (int i = 0; i < num_segments; i++) {
            int segment_size = (i == num_segments - 1) ? (msg_bytes - i * burst_bytes) : burst_bytes;
            int seg_count = segment_size / type_size;
            int offset = i * burst_bytes;

            if (i != num_segments - 1) {
                MPI_Request* seg_req_ptr = (MPI_Request*)malloc(sizeof(MPI_Request));
                if (!seg_req_ptr) return MPI_ERR_OTHER;

                err = mpi_isend_seg((char*)buf + offset, seg_count, datatype,
                                    dest, tag + SEGMENT_TAG_OFFSET + i, comm, seg_req_ptr);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Isend.\n", err);
                    return err;
                }

                info->allocated_requests[i] = seg_req_ptr;
                info->requests[i] = *seg_req_ptr;

            } else {
                err = mpi_isend_seg((char*)buf + offset, seg_count, datatype,
                                    dest, tag + SEGMENT_TAG_OFFSET + i, comm, request);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Isend.\n", err);
                    return err;
                }

                info->requests[i] = *request;
                add_seg_info(request, info);
            }
        }
    } else {
        return mpi_isend_seg(buf, count, datatype, dest, tag, comm, request);
    }

    return err;
}

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request* request) {
    int type_size = 0, err = 0;
    PMPI_Type_size(datatype, &type_size);
    size_t msg_bytes = (size_t)count * (size_t)type_size;

    if (msg_bytes > burst_bytes) {
        int num_segments = (msg_bytes + burst_bytes - 1) / burst_bytes;

        SegInfo* seginfo = (SegInfo*)calloc(1, sizeof(SegInfo));
        if (!seginfo) return MPI_ERR_OTHER;
        seginfo->type = RECV;
        seginfo->segmented = 1;
        seginfo->num_segments = num_segments;
        seginfo->requests = (MPI_Request*)malloc(sizeof(MPI_Request) * num_segments);
        seginfo->allocated_requests = (MPI_Request**)malloc(sizeof(MPI_Request*) * (num_segments - 1));
        if (!seginfo->requests || !seginfo->allocated_requests) return MPI_ERR_OTHER;

        for (int i = 0; i < num_segments; i++) {
            int segment_size = (i == num_segments - 1) ? (msg_bytes - i * burst_bytes) : burst_bytes;
            int seg_count = segment_size / type_size;
            int offset = i * burst_bytes;
            int seg_tag = tag + SEGMENT_TAG_OFFSET + i;

            if (i != num_segments - 1) {
                MPI_Request* seg_req_ptr = (MPI_Request*)malloc(sizeof(MPI_Request));
                if (!seg_req_ptr) return MPI_ERR_OTHER;

                err = PMPI_Irecv((char*)buf + offset, seg_count, datatype,
                                 source, seg_tag, comm, seg_req_ptr);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Irecv.\n", err);
                    return err;
                }

                seginfo->requests[i] = *seg_req_ptr;
                seginfo->allocated_requests[i] = seg_req_ptr;

                RequestInfo* info = (RequestInfo*)calloc(1, sizeof(RequestInfo));
                if (!info) return MPI_ERR_OTHER;

                info->type = RECV;
                info->pipeline = 0;
                info->source = source;
                info->user_buffer = (char*)buf + offset;
                info->buffer_size = segment_size;
                info->tag = seg_tag;
                info->num_requests = 1;
                info->requests = (MPI_Request*)malloc(sizeof(MPI_Request));
                if (!info->requests) return MPI_ERR_OTHER;
                info->requests[0] = *seg_req_ptr;

                add_request_info(seg_req_ptr, info);

            } else {
                err = PMPI_Irecv((char*)buf + offset, seg_count, datatype,
                                 source, seg_tag, comm, request);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Irecv.\n", err);
                    return err;
                }

                seginfo->requests[i] = *request;
                add_seg_info(request, seginfo);

                RequestInfo* info = (RequestInfo*)calloc(1, sizeof(RequestInfo));
                if (!info) return MPI_ERR_OTHER;

                info->type = RECV;
                info->pipeline = 0;
                info->source = source;
                info->user_buffer = (char*)buf + offset;
                info->buffer_size = segment_size;
                info->tag = seg_tag;
                info->num_requests = 1;
                info->requests = (MPI_Request*)malloc(sizeof(MPI_Request));
                if (!info->requests) return MPI_ERR_OTHER;
                info->requests[0] = *request;

                add_request_info(request, info);
            }
        }

    } else {
        err = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
        if (err != MPI_SUCCESS) {
            printf("Error %d in MPI_Irecv.\n", err);
            return err;
        }

        RequestInfo* info = (RequestInfo*)calloc(1, sizeof(RequestInfo));
        if (!info) return MPI_ERR_OTHER;

        info->type = RECV;
        info->pipeline = 0;
        info->source = source;
        info->user_buffer = buf;
        info->buffer_size = (int)msg_bytes;
        info->tag = tag;
        info->num_requests = 1;
        info->requests = (MPI_Request*)malloc(sizeof(MPI_Request));
        if (!info->requests) return MPI_ERR_OTHER;
        info->requests[0] = *request;

        add_request_info(request, info);
    }

    return err;
}

int mpi_wait_seg(MPI_Request* request, MPI_Status* status) {
    int err = MPI_SUCCESS;
    if (request == NULL) {
        printf("[Error] request is null.\n");
        return MPI_ERR_REQUEST;
    }

    RequestInfo* info = find_request_info(request);

    if (status == MPI_STATUS_IGNORE) {
        err = PMPI_Wait(request, MPI_STATUS_IGNORE);
    } else {
        err = PMPI_Wait(request, status);
        if (err != MPI_SUCCESS) {
            printf("Error %d in MPI_Wait.\n", err);
            return err;
        }
    }

    if (info) {
        if (info->type == SEND) {
            if (info->pipeline) {
                for (int j = 1; j < info->num_requests; j++) {
                    MPI_Status seg_status;
                    err = PMPI_Wait(&info->requests[j], &seg_status);
                    if (err != MPI_SUCCESS) {
                        return err;
                    }
                }

                for (int j = 0; j < info->num_send_buffers; j++) {
                    free(info->send_buffers[j]);
                }
            } else {
                for (int j = 0; j < info->num_send_buffers; j++) {
                    free(info->send_buffers[j]);
                }
            }
        } else if (info->type == RECV) {
        } else {
            printf("Unknown request type.\n");
        }

        remove_request_info(request);
    } else {
        printf("Request info not found.\n");
    }

    return err;
}

int MPI_Wait(MPI_Request* request, MPI_Status* status) {
    SegInfo* seginfo = find_seg_info(request);
    if (seginfo) {
        int err = MPI_SUCCESS;

        for (int i = 0; i < seginfo->num_segments; i++) {
            MPI_Status seg_status;

            if (i != seginfo->num_segments - 1) {
                MPI_Request* seg_req_ptr = seginfo->allocated_requests[i];
                err = mpi_wait_seg(seg_req_ptr, &seg_status);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Wait (segment %d).\n", err, i);
                    return err;
                }
            } else {
                err = mpi_wait_seg(request, &seg_status);
                if (err != MPI_SUCCESS) {
                    printf("Error %d in MPI_Wait (final segment).\n", err);
                    return err;
                }
            }
        }

        for (int i = 0; i < seginfo->num_segments - 1; i++) {
            free(seginfo->allocated_requests[i]);
        }

        free(seginfo->allocated_requests);
        free(seginfo->requests);

        remove_seg_info(request);
        return err;

    } else {
        return mpi_wait_seg(request, status);
    }
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    int err = MPI_SUCCESS;

    if (array_of_requests == NULL) {
        return MPI_Wait(NULL, NULL);
    }

    for (int i = 0; i < count; i++) {
        if (array_of_requests[i] == MPI_REQUEST_NULL) {
            continue;
        }

        err = MPI_Wait(
            &array_of_requests[i],
            (array_of_statuses == MPI_STATUSES_IGNORE) ? MPI_STATUS_IGNORE : &array_of_statuses[i]
        );

        if (err != MPI_SUCCESS) {
            return err;
        }
    }

    return err;
}

int MPI_Send(const void* buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
    int err = MPI_SUCCESS;
    MPI_Request request;

    err = MPI_Isend(buf, count, datatype, dest, tag, comm, &request);
    if (err != MPI_SUCCESS) return err;

    err = MPI_Wait(&request, MPI_STATUS_IGNORE);
    return err;
}

int MPI_Recv(void* buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int err = MPI_SUCCESS;
    MPI_Request request;

    err = MPI_Irecv(buf, count, datatype, source, tag, comm, &request);
    if (err != MPI_SUCCESS) return err;

    err = MPI_Wait(&request, status);
    return err;
}

/* Implementation of MPI_Sendrecv using Irecv + Isend + Wait(Isend) + Wait(Irecv) */
int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm,
                 MPI_Status* status) {
    int mpi_errno;
    MPI_Request req_send, req_recv;

    mpi_errno = MPI_Irecv(recvbuf, recvcount, recvtype, source, recvtag, comm, &req_recv);
    if (mpi_errno != MPI_SUCCESS)
        return mpi_errno;

    mpi_errno = MPI_Isend(sendbuf, sendcount, sendtype, dest, sendtag, comm, &req_send);
    if (mpi_errno != MPI_SUCCESS) {
        PMPI_Cancel(&req_recv);
        return mpi_errno;
    }

    mpi_errno = MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS) {
        return mpi_errno;
    }

    mpi_errno = MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    if (mpi_errno != MPI_SUCCESS) {
        return mpi_errno;
    }

    return MPI_SUCCESS;
}

int MPIR_Allgatherv_intra_ring(const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const int * recvcounts,
    const int * displs,
    MPI_Datatype recvtype,
    MPI_Comm comm)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    MPI_Aint total_count;
    recvtype_extent = sizeof(recvtype);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;
    torecv = total_count - recvcounts[rank];
    tosend = total_count - recvcounts[right];

    chunk_count = 0;
    max = recvcounts[0];
    for (i = 1; i < comm_size; i++)
        if (max < recvcounts[i])
            max = recvcounts[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0 &&
        max * recvtype_extent > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE) {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE / recvtype_extent;
        if (!chunk_count)
            chunk_count = 1;
    }
    if (!chunk_count)
        chunk_count = max;

    MPI_Aint soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    while (tosend || torecv) {
        MPI_Aint sendnow, recvnow;
        sendnow = ((recvcounts[sidx] - soffset) >
                   chunk_count) ? chunk_count : (recvcounts[sidx] - soffset);
        recvnow = ((recvcounts[ridx] - roffset) >
                   chunk_count) ? chunk_count : (recvcounts[ridx] - roffset);

        char *sbuf, *rbuf;
        sbuf = (char *) recvbuf + ((displs[sidx] + soffset) * recvtype_extent);
        rbuf = (char *) recvbuf + ((displs[ridx] + roffset) * recvtype_extent);

        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
        } else if (!sendnow) {
            mpi_errno = MPI_Recv(rbuf, recvnow, recvtype, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            torecv -= recvnow;
        } else if (!recvnow) {
            mpi_errno = MPI_Send(sbuf, sendnow, recvtype, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
        } else {
            mpi_errno = MPI_Sendrecv(sbuf,
                sendnow,
                recvtype,
                right,
                MPIR_ALLGATHERV_TAG,
                rbuf,
                recvnow,
                recvtype,
                left,
                MPIR_ALLGATHERV_TAG,
                comm,
                &status);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }

        soffset += sendnow;
        roffset += recvnow;
        if (soffset == recvcounts[sidx]) {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == recvcounts[ridx]) {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPI_Finalize(void) {
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);

    int nodeRank;
    MPI_Comm_rank(nodeComm, &nodeRank);

    double localStart_us, localEnd_us;
    if (sendEventCount > 0) {
        localStart_us = sendEvents[0].timestamp;
        localEnd_us = sendEvents[sendEventCount - 1].timestamp;
    } else {
        localStart_us = DBL_MAX;
        localEnd_us = 0.0;
    }

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    hostname[sizeof(hostname) - 1] = '\0';
    char* dot = strchr(hostname, '.');
    if (dot) *dot = '\0';

    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long time_now = (long long)tv.tv_sec * 1000000 + tv.tv_usec;

    char* env = getenv("MAX_RATE");
    char limiter[64];
    if (env) {
        strncpy(limiter, env, sizeof(limiter));
    } else {
        snprintf(limiter, sizeof(limiter), "%.0f", BANDWIDTH_LIMIT);
    }

    // pid_t pid = getpid();

    // char filename[512];
    // snprintf(filename, sizeof(filename), "./sendEvents-topseg/sendEvents_%s_%lld_%d%s%d.txt",
    //          limiter, time_now, pid, hostname, globalRank);

    // FILE* fp = fopen(filename, "w");
    // if (fp) {
    //     for (int i = 0; i < sendEventCount; i++) {
    //         fprintf(fp, "%.0f %zu\n", sendEvents[i].timestamp, sendEvents[i].size);
    //     }
    //     fclose(fp);
    // } else {
    //     fprintf(stderr, "Error opening file: %s\n", filename);
    // }

    // Global time range
    double globalStart_us, globalEnd_us;
    MPI_Allreduce(&localStart_us, &globalStart_us, 1, MPI_DOUBLE, MPI_MIN, nodeComm);
    MPI_Allreduce(&localEnd_us, &globalEnd_us, 1, MPI_DOUBLE, MPI_MAX, nodeComm);

    if (globalStart_us == DBL_MAX || globalEnd_us == 0.0) {
        if (nodeRank == 0)
            printf("No MPI_Isend events recorded on this node.\n");
        MPI_Comm_free(&nodeComm);
        return PMPI_Finalize();
    }

    double duration_us = globalEnd_us - globalStart_us;
    int numBins = (int)(ceil(duration_us / 100000.0)) + 1;

    size_t* localHistogram = (size_t*)calloc(numBins, sizeof(size_t));
    for (int i = 0; i < sendEventCount; i++) {
        int bin = (int)((sendEvents[i].timestamp - globalStart_us) / 100000.0);
        if (bin >= numBins) bin = numBins - 1;
        localHistogram[bin] += sendEvents[i].size;
    }

    if (nodeRank == 0) {
        printf("Node-level network bandwidth usage (in bytes) per 0.1s interval:\n");
        fflush(stdout);
        for (int i = 0; i < numBins; i++) {
            double intervalStart_us = globalStart_us + i * 100000.0;
            double intervalEnd_us = intervalStart_us + 100000.0;

            time_t start_sec = (time_t)(intervalStart_us / 1e6);
            long start_ms = (long)(fmod(intervalStart_us, 1e6) / 1e3);

            time_t end_sec = (time_t)(intervalEnd_us / 1e6);
            long end_ms = (long)(fmod(intervalEnd_us, 1e6) / 1e3);

            struct tm* start_tm = localtime(&start_sec);
            struct tm* end_tm = localtime(&end_sec);

            char start_str[64], end_str[64];
            strftime(start_str, sizeof(start_str), "%a %b %d %H:%M:%S", start_tm);
            strftime(end_str, sizeof(end_str), "%H:%M:%S", end_tm);

            double Tput_Gbps = (double)localHistogram[i] * 8.0 / 1024.0 / 1024.0 / 1024.0 / 0.1;

            printf("[%s:%03ld - %s:%03ld): %.3f Gbps\n", start_str, start_ms, end_str, end_ms, Tput_Gbps);
            fflush(stdout);
        }
    }

    free(localHistogram);
    MPI_Comm_free(&nodeComm);
    return PMPI_Finalize();
}

/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

int MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    unsigned char *outputBytes,
    float compressionRatio,
    float tolerance,
    int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;
    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));

    void *temp_recvbuf = (void *) malloc(total_count * recvtype_extent);

    osbuf = (char *) recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    ZCCL_float_openmp_threadblock_arg(
        sbuf, osbuf, &outSize, absErrBound, (recvcounts[sidx]), blockSize);

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0
        && max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE) {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;

        if (!chunk_count)
            chunk_count = 1;
    }

    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv) {
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[ridx] - roffset);

        sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *) temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
        } else if (!sendnow) {
            mpi_errno = MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            torecv -= recvnow;
        } else if (!recvnow) {
            mpi_errno = MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }

            tosend -= sendnow;
        } else {
            mpi_errno = MPI_Sendrecv(sbuf,
                sendnow,
                MPI_BYTE,
                right,
                MPIR_ALLGATHERV_TAG,
                rbuf,
                recvnow,
                MPI_BYTE,
                left,
                MPIR_ALLGATHERV_TAG,
                comm,
                &status);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx]) {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx]) {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++) {
        ZCCL_float_decompress_openmp_threadblock_arg((char *) recvbuf + displs[i] * recvtype_extent,
            recvcounts[i],
            absErrBound,
            blockSize,
            (char *) temp_recvbuf + displs[i] * recvtype_extent);
    }
    free(temp_recvbuf);
    free(compressed_sizes);
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPI_Allreduce_ZCCL_RI2_mt_oa_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS, mpi_errno_ret = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    size_t extent;
    MPI_Aint lb, true_extent;
    MPI_Aint *cnts, *displs;
    int send_rank, recv_rank, total_count;
    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = count;
    void *tmpbuf;
    int tag;
    int flag;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    extent = sizeof(datatype);
    is_inplace = (sendbuf == MPI_IN_PLACE);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    size_t outSize;
    size_t byteLength;

    cnts = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(cnts != NULL);
    displs = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(displs != NULL);

    for (i = 0; i < nranks; i++)
        cnts[i] = 0;

    total_count = 0;
    for (i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    if (!is_inplace) {
        memcpy(recvbuf, sendbuf, sizeof(datatype) * count);
    }

    tmpbuf = (void *) malloc(tmp_len * extent);

    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * extent);
    float *newData = (float *) malloc(cnts[0] * extent);

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++) {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

        tag = tag_base;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank] * extent, MPI_BYTE, src, tag, comm, &reqs[0]);
        if (mpi_errno) {
            exit(-1);
        }
        MPI_Test(&reqs[0], &flag, &stas[0]);
        ZCCL_float_openmp_threadblock_arg(outputBytes,
            (char *) recvbuf + displs[send_rank] * extent,
            &outSize,
            absErrBound,
            cnts[send_rank],
            blockSize);
        unsigned char *bytes = outputBytes;
        mpi_errno = MPI_Isend(bytes, outSize, MPI_BYTE, dst, tag, comm, &reqs[1]);
        if (mpi_errno) {
            exit(-1);
        }
        if (mpi_errno) {
            exit(-1);
        }
        MPI_Wait(&reqs[0], &stas[0]);

        MPI_Test(&reqs[1], &flag, &stas[1]);
        ZCCL_float_decompress_openmp_threadblock_arg(
            newData, cnts[recv_rank], absErrBound, blockSize, tmpbuf);
        mpi_errno = MPI_Reduce_local(
            newData, (char *) recvbuf + displs[recv_rank] * extent, cnts[recv_rank], datatype, op);

        MPI_Wait(&reqs[1], &stas[1]);
        if (mpi_errno) {
            exit(-1);
        }
    }

    mpi_errno = MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(MPI_IN_PLACE,
        -1,
        MPI_DATATYPE_NULL,
        recvbuf,
        cnts,
        displs,
        datatype,
        comm,
        outputBytes,
        compressionRatio,
        tolerance,
        blockSize);
    if (mpi_errno) {
        exit(-1);
    }

    free(outputBytes);
    free(cnts);
    free(displs);
    free(tmpbuf);
    free(newData);

    return mpi_errno;
}

int MPIR_Allgatherv_intra_ring_RI2_st_oa_record(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    unsigned char *outputBytes,
    float compressionRatio,
    float tolerance,
    int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;

    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));

    void *temp_recvbuf = (void *) malloc(total_count * recvtype_extent);

    osbuf = (char *) recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    ZCCL_float_single_thread_arg(sbuf, osbuf, &outSize, absErrBound, (recvcounts[sidx]), blockSize);

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0
        && max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE) {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;

        if (!chunk_count)
            chunk_count = 1;
    }

    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv) {
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[ridx] - roffset);

        sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *) temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
        } else if (!sendnow) {
            mpi_errno = MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            torecv -= recvnow;
        } else if (!recvnow) {
            mpi_errno = MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
        } else {
            mpi_errno = MPI_Sendrecv(sbuf,
                sendnow,
                MPI_BYTE,
                right,
                MPIR_ALLGATHERV_TAG,
                rbuf,
                recvnow,
                MPI_BYTE,
                left,
                MPIR_ALLGATHERV_TAG,
                comm,
                &status);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx]) {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx]) {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++) {
        ZCCL_float_decompress_single_thread_arg((char *) recvbuf + displs[i] * recvtype_extent,
            recvcounts[i],
            absErrBound,
            blockSize,
            (char *) temp_recvbuf + displs[i] * recvtype_extent);
    }
    free(temp_recvbuf);
    free(compressed_sizes);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPI_Allreduce_ZCCL_RI2_st_oa_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS, mpi_errno_ret = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    size_t extent;
    MPI_Aint lb, true_extent;
    MPI_Aint *cnts, *displs;
    int send_rank, recv_rank, total_count;
    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = count;
    void *tmpbuf;
    int tag;
    int flag;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    extent = sizeof(datatype);
    is_inplace = (sendbuf == MPI_IN_PLACE);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    size_t outSize;
    size_t byteLength;

    cnts = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(cnts != NULL);
    displs = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(displs != NULL);

    for (i = 0; i < nranks; i++)
        cnts[i] = 0;

    total_count = 0;
    for (i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    if (!is_inplace) {
        memcpy(recvbuf, sendbuf, sizeof(datatype) * count);
    }

    tmpbuf = (void *) malloc(tmp_len * extent);

    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * extent);
    float *newData = (float *) malloc(cnts[0] * extent);

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++) {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

        tag = tag_base;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank] * extent, MPI_BYTE, src, tag, comm, &reqs[0]);
        if (mpi_errno) {
            exit(-1);
        }
        MPI_Test(&reqs[0], &flag, &stas[0]);
        ZCCL_float_single_thread_arg(outputBytes,
            (char *) recvbuf + displs[send_rank] * extent,
            &outSize,
            absErrBound,
            cnts[send_rank],
            blockSize);
        unsigned char *bytes = outputBytes;
        mpi_errno = MPI_Isend(bytes, outSize, MPI_BYTE, dst, tag, comm, &reqs[1]);
        if (mpi_errno) {
            exit(-1);
        }
        if (mpi_errno) {
            exit(-1);
        }
        MPI_Wait(&reqs[0], &stas[0]);

        MPI_Test(&reqs[1], &flag, &stas[1]);
        ZCCL_float_decompress_single_thread_arg(
            newData, cnts[recv_rank], absErrBound, blockSize, tmpbuf);
        mpi_errno = MPI_Reduce_local(
            newData, (char *) recvbuf + displs[recv_rank] * extent, cnts[recv_rank], datatype, op);

        MPI_Wait(&reqs[1], &stas[1]);
        if (mpi_errno) {
            exit(-1);
        }
    }

    mpi_errno = MPIR_Allgatherv_intra_ring_RI2_st_oa_record(MPI_IN_PLACE,
        -1,
        MPI_DATATYPE_NULL,
        recvbuf,
        cnts,
        displs,
        datatype,
        comm,
        outputBytes,
        compressionRatio,
        tolerance,
        blockSize);
    if (mpi_errno) {
        exit(-1);
    }

    free(outputBytes);
    free(cnts);
    free(displs);
    free(tmpbuf);
    free(newData);

    return mpi_errno;
}

int MPI_Allreduce_ZCCL_RI2_st_oa_op_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS, mpi_errno_ret = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    size_t extent;
    MPI_Aint lb, true_extent;
    MPI_Aint *cnts, *displs;
    int send_rank, recv_rank, total_count;
    int tmp_len = 0;
    double absErrBound = compressionRatio;

    tmp_len = count;

    void *tmpbuf;
    int tag;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    extent = sizeof(datatype);
    is_inplace = (sendbuf == MPI_IN_PLACE);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    size_t outSize;
    size_t byteLength;

    cnts = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(cnts != NULL);
    displs = (MPI_Aint *) malloc(nranks * sizeof(MPI_Aint));
    assert(displs != NULL);

    for (i = 0; i < nranks; i++)
        cnts[i] = 0;

    total_count = 0;
    for (i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    if (!is_inplace) {
        memcpy(recvbuf, sendbuf, sizeof(datatype) * count);
    }

    tmpbuf = (void *) malloc(tmp_len * extent);
    int chunk_size;
    chunk_size = SINGLETHREAD_CHUNK_SIZE;
    int chunk_num = cnts[0] / chunk_size;
    unsigned char *outputBytes =
        (unsigned char *) malloc(tmp_len * extent + sizeof(size_t) * (chunk_num + 1));
    float *newData = (float *) malloc(cnts[0] * extent);

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++) {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

        tag = tag_base;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank] * extent, MPI_BYTE, src, tag, comm, &reqs[0]);
        if (mpi_errno) {
            exit(-1);
        }

        outSize = 0;
        chunk_num = cnts[send_rank] / chunk_size;
        int chunk_remainder_size = cnts[send_rank] % chunk_size;
        int flag, iter;
        size_t chunk_out_size = 0;

        for (iter = 0; iter < chunk_num; iter++) {
            MPI_Test(&reqs[0], &flag, &stas[0]);
            ZCCL_float_single_thread_arg_split_record(
                outputBytes + outSize + sizeof(size_t) * (chunk_num + 1),
                (char *) recvbuf + displs[send_rank] * extent + iter * chunk_size * extent,
                &chunk_out_size,
                absErrBound,
                chunk_size,
                blockSize,
                outputBytes,
                iter);
            outSize += chunk_out_size;
        }
        if (chunk_remainder_size != 0) {
            MPI_Test(&reqs[0], &flag, &stas[0]);

            ZCCL_float_single_thread_arg_split_record(
                outputBytes + outSize + sizeof(size_t) * (chunk_num + 1),
                (char *) recvbuf + displs[send_rank] * extent + iter * chunk_size * extent,
                &chunk_out_size,
                absErrBound,
                chunk_remainder_size,
                blockSize,
                outputBytes,
                iter);

            outSize += chunk_out_size;
        }
        unsigned char *bytes = outputBytes;
        mpi_errno = MPI_Isend(bytes,
            (outSize + sizeof(size_t) * (chunk_num + 1)),
            MPI_BYTE,
            dst,
            tag,
            comm,
            &reqs[1]);
        if (mpi_errno) {
            exit(-1);
        }
        MPI_Wait(&reqs[0], &stas[0]);

        outSize = 0;
        chunk_num = cnts[recv_rank] / chunk_size;
        chunk_remainder_size = cnts[recv_rank] % chunk_size;
        chunk_out_size = 0;
        int decom_offset = 0;
        int decom_out_offset = 0;
        size_t *chunk_arr = (size_t *) tmpbuf;
        unsigned char *cmpBytes;
        for (iter = 0; iter < chunk_num; iter++) {
            MPI_Test(&reqs[1], &flag, &stas[1]);
            cmpBytes = (unsigned char *) tmpbuf + sizeof(size_t) * (chunk_num + 1) + decom_offset;
            ZCCL_float_decompress_single_thread_arg((unsigned char *) newData + decom_out_offset,
                chunk_size,
                absErrBound,
                blockSize,
                cmpBytes);
            decom_offset += chunk_arr[iter];
            decom_out_offset += chunk_size * extent;
        }
        if (chunk_remainder_size != 0) {
            MPI_Test(&reqs[1], &flag, &stas[1]);
            cmpBytes = (unsigned char *) tmpbuf + sizeof(size_t) * (chunk_num + 1) + decom_offset;
            ZCCL_float_decompress_single_thread_arg((unsigned char *) newData + decom_out_offset,
                chunk_remainder_size,
                absErrBound,
                blockSize,
                cmpBytes);
        }
        mpi_errno = MPI_Reduce_local(
            newData, (char *) recvbuf + displs[recv_rank] * extent, cnts[recv_rank], datatype, op);
        MPI_Wait(&reqs[1], &stas[1]);
        if (mpi_errno) {
            exit(-1);
        }
    }

    mpi_errno = MPIR_Allgatherv_intra_ring_RI2_st_oa_record(MPI_IN_PLACE,
        -1,
        MPI_DATATYPE_NULL,
        recvbuf,
        cnts,
        displs,
        datatype,
        comm,
        outputBytes,
        compressionRatio,
        tolerance,
        blockSize);
    if (mpi_errno) {
        exit(-1);
    }

    free(outputBytes);
    free(newData);
    free(cnts);
    free(displs);
    free(tmpbuf);
    return mpi_errno;
}

/**
 * Pipelined & Multidimensional data scheme
*/

int MPIR_Allgatherv_intra_ring_RI2_mt_oa_record_seg(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    unsigned char *outputBytes,
    float compressionRatio,
    float tolerance,
    int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    int recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    int send_num_seg = (recvcounts[0] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    int recv_num_seg = (recvcounts[0] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    int halfpoint = send_num_seg;
    int *send_seg_cnts = (int *) malloc(send_num_seg * sizeof(int));
    int *send_seg_displs = (int *) malloc(send_num_seg * sizeof(int));
    int *recv_seg_cnts = (int *) malloc(recv_num_seg * sizeof(int));
    int *recv_seg_displs = (int *) malloc(recv_num_seg * sizeof(int));
    MPI_Request *send_reqs = (MPI_Request *) malloc(send_num_seg * sizeof(MPI_Request));
    MPI_Request *recv_reqs = (MPI_Request *) malloc(recv_num_seg * sizeof(MPI_Request));
    MPI_Status *send_stas = (MPI_Status *) malloc(send_num_seg * sizeof(MPI_Status));
    MPI_Status *recv_stas = (MPI_Status *) malloc(recv_num_seg * sizeof(MPI_Status));
    int *send_tags = (int *) malloc(send_num_seg * sizeof(int));
    int *recv_tags = (int *) malloc(recv_num_seg * sizeof(int));
    bool *recv_record = (bool *) malloc(recv_num_seg * sizeof(bool));
    unsigned char **bytes = malloc(2 * send_num_seg * sizeof(unsigned char *));
    int initial_send_num_seg = 2 * send_num_seg;
    for (i = 0; i < initial_send_num_seg; i++){
        bytes[i] = malloc(2 * SEGSIZE); // in case of compression inflation
    }
    for (i = 0; i < send_num_seg; i++){
        send_tags[i] = recv_tags[i] = tag_base + i;
    }

    // Segment-level compression
    send_num_seg = (recvcounts[rank] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    for (i = 0; i < send_num_seg; ++i) {
        send_seg_displs[i] = i * SEGSIZE;
        send_seg_cnts[i] = (i < send_num_seg - 1) ? SEGSIZE : (recvcounts[rank] * recvtype_extent - i * SEGSIZE);
    }
    for (i = 0; i < send_num_seg; i++){
        ZCCL_float_openmp_threadblock_arg(bytes[i] + sizeof(size_t), (unsigned char *) recvbuf + displs[rank] * recvtype_extent + send_seg_displs[i], &outSize, absErrBound, send_seg_cnts[i] / recvtype_extent, blockSize);
        assert(outSize <= 2 * SEGSIZE);
        memcpy(bytes[i], &outSize, sizeof(size_t));
    }

    // Communication & decompression overlap
    int j, send_index, recv_index;
    int sidx, ridx;
    int send_count, recv_count;
    int flag = 0;
    sidx = rank;
    ridx = left;
    
    for (i = 0; i < comm_size - 1; i++){
        // printf("From rank %d, sidx = %d, ridx = %d\n", rank, sidx, ridx);
        send_index = i % 2 * halfpoint;
        recv_index = (i + 1) % 2 * halfpoint;

        send_num_seg = (recvcounts[sidx] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
        recv_num_seg = (recvcounts[ridx] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
        // printf("From rank %d, send_num_seg = %d, recv_num_seg = %d\n", rank, send_num_seg, recv_num_seg);
        fflush(stdout);
        for (j = 0; j < send_num_seg; ++j) {
            send_seg_displs[j] = j * SEGSIZE;
            send_seg_cnts[j] = (j < send_num_seg - 1) ? SEGSIZE : (recvcounts[sidx] * recvtype_extent - j * SEGSIZE);
            // printf("rank %d, send_seg_cnts[%d] to rank %d = %d\n", rank, j, right, send_seg_cnts[j]);
        }
        for (j = 0; j < recv_num_seg; ++j) {
            recv_seg_displs[j] = j * SEGSIZE;
            recv_seg_cnts[j] = (j < recv_num_seg - 1) ? SEGSIZE : (recvcounts[ridx] * recvtype_extent - j * SEGSIZE);
            // printf("rank %d, recv_seg_cnts[%d] from rank %d = %d\n", rank, j, left, recv_seg_cnts[j]);
            // printf("recv_seg_cnts[%d] = %d\n", j, recv_seg_cnts[j]);
        }
        memset(recv_record, 0, sizeof(bool) * recv_num_seg);
        send_count = 0;
        recv_count = 0;
        for (j = 0; j < recv_num_seg; j++){
            mpi_errno = MPI_Irecv(bytes[recv_index + j], recv_seg_cnts[j], MPI_BYTE, left, recv_tags[j], comm, &recv_reqs[j]);
            if (mpi_errno){
                exit(-1);
            }
        }
        // printf("From rank %d, iteration %d, Irecv hanged\n", rank, i);
        while(send_count < send_num_seg){
            for (j = 0; j < recv_num_seg; j++){
                if (!recv_record[j]) {
                    MPI_Test(&recv_reqs[j], &flag, &recv_stas[j]);
                    if (flag) {
                        recv_count++;
                        recv_record[j] = true;
                        memcpy(&outSize, bytes[recv_index + j], sizeof(size_t));
                        // printf("iter %d, rank %d receive seg %d from rank %d with size %d\n", i, rank, j, left, outSize);
                        // fflush(stdout);
                        // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] decompress to %p\n", i, rank, ridx * recv_num_seg + j, recv_index + j, (unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j]);
                        ZCCL_float_decompress_openmp_threadblock_arg((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
                                        recv_seg_cnts[j] / recvtype_extent, absErrBound, blockSize,
                                        bytes[recv_index + j] + sizeof(size_t));
                        flag = 0;
                        break;
                    }
                }
            }
            memcpy(&outSize, bytes[send_index + send_count], sizeof(size_t));
            // printf("iter %d, rank %d send seg %d to rank %d with size %d\n", i, rank, send_count, right, outSize);
            // fflush(stdout);
            // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] send to Rank %d\n", i, rank, sidx * send_num_seg + send_count, send_index + send_count, right);
            mpi_errno = MPI_Isend(bytes[send_index + send_count], outSize + sizeof(size_t), MPI_BYTE, right, send_tags[send_count], comm, &send_reqs[send_count]);
            if (mpi_errno) {
                exit(-1);
            }
            send_count++;
        }
        // printf("From rank %d, iteration %d, Isend hanged\n", rank, i);
        MPI_Waitall(send_num_seg, send_reqs, send_stas);
        if (recv_count < recv_num_seg) {
            for (j = 0; j < recv_num_seg; j++){
                if(!recv_record[j]) {
                    MPI_Wait(&recv_reqs[j], &recv_stas[j]);
                    memcpy(&outSize, bytes[recv_index + j], sizeof(size_t));
                    // printf("iter %d, rank %d receive seg %d from rank %d with size %d\n", i, rank, j, left, outSize);
                    // fflush(stdout);
                    // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] decompress to %p\n", i, rank, ridx * recv_num_seg + j, recv_index + j, (unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j]);
                    ZCCL_float_decompress_openmp_threadblock_arg((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
                                    recv_seg_cnts[j] / recvtype_extent, absErrBound, blockSize,
                                    bytes[recv_index + j] + sizeof(size_t));
                }
            }
        }
        
        sidx = (sidx + comm_size - 1) % comm_size;
        ridx = (ridx + comm_size - 1) % comm_size;
    }
    for (i = 0; i < initial_send_num_seg; i++){
        free(bytes[i]);
    }
    free(bytes);
    free(send_seg_cnts);
    free(send_seg_displs);
    free(recv_seg_cnts);
    free(recv_seg_displs);
    free(send_reqs);
    free(recv_reqs);
    free(send_stas);
    free(recv_stas);
    free(send_tags);
    free(recv_tags);
    free(recv_record);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPIR_Allgatherv_intra_ring_RI2_st_oa_record_seg(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    unsigned char *outputBytes,
    float compressionRatio,
    float tolerance,
    int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    int send_num_seg = (recvcounts[0] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    int recv_num_seg = (recvcounts[0] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    int halfpoint = send_num_seg;
    MPI_Aint *send_seg_cnts = (MPI_Aint *) malloc(send_num_seg * sizeof(MPI_Aint));
    MPI_Aint *send_seg_displs = (MPI_Aint *) malloc(send_num_seg * sizeof(MPI_Aint));
    MPI_Aint *recv_seg_cnts = (MPI_Aint *) malloc(recv_num_seg * sizeof(MPI_Aint));
    MPI_Aint *recv_seg_displs = (MPI_Aint *) malloc(recv_num_seg * sizeof(MPI_Aint));
    MPI_Request *send_reqs = (MPI_Request *) malloc(send_num_seg * sizeof(MPI_Request));
    MPI_Request *recv_reqs = (MPI_Request *) malloc(recv_num_seg * sizeof(MPI_Request));
    MPI_Status *send_stas = (MPI_Status *) malloc(send_num_seg * sizeof(MPI_Status));
    MPI_Status *recv_stas = (MPI_Status *) malloc(recv_num_seg * sizeof(MPI_Status));
    int *send_tags = (int *) malloc(send_num_seg * sizeof(int));
    int *recv_tags = (int *) malloc(recv_num_seg * sizeof(int));
    bool *recv_record = (bool *) malloc(recv_num_seg * sizeof(bool));
    unsigned char **bytes = malloc(2 * send_num_seg * sizeof(unsigned char *));
    int initial_send_num_seg = 2 * send_num_seg;
    for (i = 0; i < initial_send_num_seg; i++){
        bytes[i] = malloc(2 * SEGSIZE); // in case of compression inflation
    }
    for (i = 0; i < send_num_seg; i++){
        send_tags[i] = recv_tags[i] = tag_base + i;
    }

    // Segment-level compression
    send_num_seg = (recvcounts[rank] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
    for (i = 0; i < send_num_seg; ++i) {
        send_seg_displs[i] = i * SEGSIZE;
        send_seg_cnts[i] = (i < send_num_seg - 1) ? SEGSIZE : (recvcounts[rank] * recvtype_extent - i * SEGSIZE);
    }
    for (i = 0; i < send_num_seg; i++){
        // zccl_timer_logf("From Allgatherv, compress seg %d starts", i);
        ZCCL_float_single_thread_arg(bytes[i] + sizeof(size_t), (unsigned char *) recvbuf + displs[rank] * recvtype_extent + send_seg_displs[i], &outSize, absErrBound, send_seg_cnts[i] / recvtype_extent, blockSize);
        memcpy(bytes[i], &outSize, sizeof(size_t));
        // zccl_timer_logf("From Allgatherv, compress seg %d ends", i);
    }

    // Communication & decompression overlap
    int j, send_index, recv_index;
    int sidx, ridx;
    int send_count, recv_count;
    int flag = 0;
    sidx = rank;
    ridx = left;
    
    for (i = 0; i < comm_size - 1; i++){
        // printf("From rank %d, sidx = %d, ridx = %d\n", rank, sidx, ridx);
        send_index = i % 2 * halfpoint;
        recv_index = (i + 1) % 2 * halfpoint;

        send_num_seg = (recvcounts[sidx] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
        recv_num_seg = (recvcounts[ridx] * recvtype_extent + SEGSIZE - 1) / SEGSIZE;
        // printf("From rank %d, send_num_seg = %d, recv_num_seg = %d\n", rank, send_num_seg, recv_num_seg);
        fflush(stdout);
        for (j = 0; j < send_num_seg; ++j) {
            send_seg_displs[j] = j * SEGSIZE;
            send_seg_cnts[j] = (j < send_num_seg - 1) ? SEGSIZE : (recvcounts[sidx] * recvtype_extent - j * SEGSIZE);
            // printf("rank %d, send_seg_cnts[%d] to rank %d = %d\n", rank, j, right, send_seg_cnts[j]);
        }
        for (j = 0; j < recv_num_seg; ++j) {
            recv_seg_displs[j] = j * SEGSIZE;
            recv_seg_cnts[j] = (j < recv_num_seg - 1) ? SEGSIZE : (recvcounts[ridx] * recvtype_extent - j * SEGSIZE);
            // printf("rank %d, recv_seg_cnts[%d] from rank %d = %d\n", rank, j, left, recv_seg_cnts[j]);
            // printf("recv_seg_cnts[%d] = %d\n", j, recv_seg_cnts[j]);
        }
        memset(recv_record, 0, sizeof(bool) * recv_num_seg);
        send_count = 0;
        recv_count = 0;
        for (j = 0; j < recv_num_seg; j++){
            // zccl_timer_logf("From Allgatherv, Irecv seg %d starts", j);
            mpi_errno = MPI_Irecv(bytes[recv_index + j], recv_seg_cnts[j], MPI_BYTE, left, recv_tags[j], comm, &recv_reqs[j]);
            if (mpi_errno){
                exit(-1);
            }
        }
        // printf("From rank %d, iteration %d, Irecv hanged\n", rank, i);
        while(send_count < send_num_seg){
            for (j = 0; j < recv_num_seg; j++){
                if (!recv_record[j]) {
                    MPI_Test(&recv_reqs[j], &flag, &recv_stas[j]);
                    if (flag) {
                        // zccl_timer_logf("From Allgatherv, Irecv seg %d ends", j);
                        recv_count++;
                        recv_record[j] = true;
                        // zccl_timer_logf("From Allgatherv, Decompress seg %d starts", j);
                        memcpy(&outSize, bytes[recv_index + j], sizeof(size_t));
                        // printf("iter %d, rank %d receive seg %d from rank %d with size %d\n", i, rank, j, left, outSize);
                        // fflush(stdout);
                        // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] decompress to %p\n", i, rank, ridx * recv_num_seg + j, recv_index + j, (unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j]);
                        ZCCL_float_decompress_single_thread_arg((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
                                        recv_seg_cnts[j] / recvtype_extent, absErrBound, blockSize,
                                        bytes[recv_index + j] + sizeof(size_t));
                        // zccl_timer_logf("From Allgatherv, Decompress seg %d ends", j);
                        flag = 0;
                        break;
                    }
                }
            }
            // zccl_timer_logf("From Allgatherv, Isend seg %d starts", send_count);
            memcpy(&outSize, bytes[send_index + send_count], sizeof(size_t));
            // printf("iter %d, rank %d send seg %d to rank %d with size %d\n", i, rank, send_count, right, outSize);
            // fflush(stdout);
            // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] send to Rank %d\n", i, rank, sidx * send_num_seg + send_count, send_index + send_count, right);
            mpi_errno = MPI_Isend(bytes[send_index + send_count], outSize + sizeof(size_t), MPI_BYTE, right, send_tags[send_count], comm, &send_reqs[send_count]);
            if (mpi_errno) {
                exit(-1);
            }
            send_count++;
        }
        // printf("From rank %d, iteration %d, Isend hanged\n", rank, i);
        MPI_Waitall(send_num_seg, send_reqs, send_stas);
        // zccl_timer_logf("From Allgatherv, Isend ends");
        if (recv_count < recv_num_seg) {
            for (j = 0; j < recv_num_seg; j++){
                if(!recv_record[j]) {
                    MPI_Wait(&recv_reqs[j], &recv_stas[j]);
                    // zccl_timer_logf("From Allgatherv, Irecv seg %d ends", j);
                    // zccl_timer_logf("From Allgatherv, Decompress seg %d starts", j);
                    memcpy(&outSize, bytes[recv_index + j], sizeof(size_t));
                    // printf("iter %d, rank %d receive seg %d from rank %d with size %d\n", i, rank, j, left, outSize);
                    // fflush(stdout);
                    // printf("Iter %d, From Rank %d, SEG %d, AKA bytes[%d] decompress to %p\n", i, rank, ridx * recv_num_seg + j, recv_index + j, (unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j]);
                    ZCCL_float_decompress_single_thread_arg((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
                                    recv_seg_cnts[j] / recvtype_extent, absErrBound, blockSize,
                                    bytes[recv_index + j] + sizeof(size_t));
                    // zccl_timer_logf("From Allgatherv, Decompress seg %d ends", j);
                }
            }
        }
        
        sidx = (sidx + comm_size - 1) % comm_size;
        ridx = (ridx + comm_size - 1) % comm_size;
    }
    for (i = 0; i < initial_send_num_seg; i++){
        free(bytes[i]);
    }
    free(bytes);
    free(send_seg_cnts);
    free(send_seg_displs);
    free(recv_seg_cnts);
    free(recv_seg_displs);
    free(send_reqs);
    free(recv_reqs);
    free(send_stas);
    free(recv_stas);
    free(send_tags);
    free(recv_tags);
    free(recv_record);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPIR_Allgatherv_intra_ring_RI2_mt_oa_3d(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    int *data_dims,
    float compressionRatio,
    float tolerance,
    int blockSideLength)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    int recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    MPI_Comm grid;
    int grid_dims[3] = {0, 0, 0};
    MPI_Dims_create(comm_size, 3, grid_dims);
    int periodic[3] = {1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, grid_dims, periodic, 0, &grid);
    int coords[3];
    MPI_Cart_coords(grid, rank, 3, coords); 

    const int n1 = data_dims[0], n2 = data_dims[1], n3 = data_dims[2];
    const int base_x = n1 / grid_dims[0], rem_x = n1 % grid_dims[0];
    const int base_y = n2 / grid_dims[1], rem_y = n2 % grid_dims[1];
    const int base_z = n3 / grid_dims[2], rem_z = n3 % grid_dims[2];
    const int bx = base_x + (coords[0] < rem_x ? 1 : 0);
    const int by = base_y + (coords[1] < rem_y ? 1 : 0);
    const int bz = base_z + (coords[2] < rem_z ? 1 : 0);
    typedef struct {
        int ox, oy, oz;
        int sx, sy, sz;
    } Meta;
    Meta *meta = (Meta*)malloc(comm_size * sizeof(Meta));
    int pref = 0, r = 0;
    int ox_acc = 0;
    for (int px = 0; px < grid_dims[0]; px++) {
        int sx = base_x + (px < rem_x ? 1 : 0);
        int ox = ox_acc;
        ox_acc += sx;
        int oy_acc = 0;
        for (int py = 0; py < grid_dims[1]; py++) {
            int sy = base_y + (py < rem_y ? 1 : 0);
            int oy = oy_acc;
            oy_acc += sy;
            int oz_acc = 0;
            for (int pz = 0; pz < grid_dims[2]; pz++) {
                int sz = base_z + (pz < rem_z ? 1 : 0);
                int oz = oz_acc;
                oz_acc += sz;
                meta[r++] = (Meta){ox, oy, oz, sx, sy, sz};
            }
        }
    }

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;

    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));

    void *temp_recvbuf = (void *) malloc(total_count * recvtype_extent);

    osbuf = (char *) recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    SZp_compress3D_fast_openmp(osbuf, sbuf, bx, by, bz, blockSideLength, absErrBound, &outSize);

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0
        && max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE) {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;

        if (!chunk_count)
            chunk_count = 1;
    }

    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv) {
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[ridx] - roffset);

        sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *) temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
        } else if (!sendnow) {
            mpi_errno = MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            torecv -= recvnow;
        } else if (!recvnow) {
            mpi_errno = MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
        } else {
            mpi_errno = MPI_Sendrecv(sbuf,
                sendnow,
                MPI_BYTE,
                right,
                MPIR_ALLGATHERV_TAG,
                rbuf,
                recvnow,
                MPI_BYTE,
                left,
                MPIR_ALLGATHERV_TAG,
                comm,
                &status);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx]) {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx]) {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++) {
        SZp_decompress3D_fast_openmp((char *) recvbuf + displs[i] * recvtype_extent,
            (char *) temp_recvbuf + displs[i] * recvtype_extent,
            meta[i].sx, meta[i].sy, meta[i].sz, 
            blockSideLength,
            absErrBound);
    }
    free(temp_recvbuf);
    free(compressed_sizes);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPIR_Allgatherv_intra_ring_RI2_st_oa_3d(const void *sendbuf,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const MPI_Aint *recvcounts,
    const MPI_Aint *displs,
    MPI_Datatype recvtype,
    MPI_Comm comm,
    int *data_dims,
    float compressionRatio,
    float tolerance,
    int blockSideLength)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    int recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    MPI_Comm grid;
    int grid_dims[3] = {0, 0, 0};
    MPI_Dims_create(comm_size, 3, grid_dims);
    int periodic[3] = {1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, grid_dims, periodic, 0, &grid);
    int coords[3];
    MPI_Cart_coords(grid, rank, 3, coords); 

    const int n1 = data_dims[0], n2 = data_dims[1], n3 = data_dims[2];
    const int base_x = n1 / grid_dims[0], rem_x = n1 % grid_dims[0];
    const int base_y = n2 / grid_dims[1], rem_y = n2 % grid_dims[1];
    const int base_z = n3 / grid_dims[2], rem_z = n3 % grid_dims[2];
    const int bx = base_x + (coords[0] < rem_x ? 1 : 0);
    const int by = base_y + (coords[1] < rem_y ? 1 : 0);
    const int bz = base_z + (coords[2] < rem_z ? 1 : 0);
    typedef struct {
        int ox, oy, oz;
        int sx, sy, sz;
    } Meta;
    Meta *meta = (Meta*)malloc(comm_size * sizeof(Meta));
    int pref = 0, r = 0;
    int ox_acc = 0;
    for (int px = 0; px < grid_dims[0]; px++) {
        int sx = base_x + (px < rem_x ? 1 : 0);
        int ox = ox_acc;
        ox_acc += sx;
        int oy_acc = 0;
        for (int py = 0; py < grid_dims[1]; py++) {
            int sy = base_y + (py < rem_y ? 1 : 0);
            int oy = oy_acc;
            oy_acc += sy;
            int oz_acc = 0;
            for (int pz = 0; pz < grid_dims[2]; pz++) {
                int sz = base_z + (pz < rem_z ? 1 : 0);
                int oz = oz_acc;
                oz_acc += sz;
                meta[r++] = (Meta){ox, oy, oz, sx, sy, sz};
            }
        }
    }

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE) {
        memcpy((char *) recvbuf + displs[rank] * sizeof(recvtype),
            sendbuf,
            sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;

    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));

    void *temp_recvbuf = (void *) malloc(total_count * recvtype_extent);

    osbuf = (char *) recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    SZp_compress3D_fast(osbuf, sbuf, bx, by, bz, blockSideLength, absErrBound, &outSize);

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0
        && max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE) {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;

        if (!chunk_count)
            chunk_count = 1;
    }

    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv) {
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) > chunk_count)
            ? chunk_count
            : (compressed_sizes[ridx] - roffset);

        sbuf = (char *) temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *) temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
        } else if (!sendnow) {
            mpi_errno = MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            torecv -= recvnow;
        } else if (!recvnow) {
            mpi_errno = MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
        } else {
            mpi_errno = MPI_Sendrecv(sbuf,
                sendnow,
                MPI_BYTE,
                right,
                MPIR_ALLGATHERV_TAG,
                rbuf,
                recvnow,
                MPI_BYTE,
                left,
                MPIR_ALLGATHERV_TAG,
                comm,
                &status);
            if (mpi_errno) {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx]) {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx]) {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++) {
        SZp_decompress3D_fast((char *) recvbuf + displs[i] * recvtype_extent,
            (char *) temp_recvbuf + displs[i] * recvtype_extent,
            meta[i].sx, meta[i].sy, meta[i].sz, 
            blockSideLength,
            absErrBound);
    }
    free(temp_recvbuf);
    free(compressed_sizes);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
