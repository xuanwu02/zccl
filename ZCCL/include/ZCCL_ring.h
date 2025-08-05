/**
 * Pacer Bandwidth Limiter Integration
 * Credit to Yuke Li @ UC Merced
*/

#ifndef ZCCL_ring_H
#define ZCCL_ring_H

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <x86intrin.h>

#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"

#define CONTROL_MSG_SIZE 16
#define SUB_MSG_TAG_OFFSET 1000000
#define SEGMENT_TAG_OFFSET 10000000

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static uint64_t BANDWIDTH_LIMIT = 10LL * 1024 * 1024 * 1000 / 8;

static int is_power_of_two(int n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}

typedef struct {
    double timestamp;
    size_t size;
} SendEvent;

#define MAX_SEND_EVENTS 10000
static SendEvent sendEvents[MAX_SEND_EVENTS];
static int sendEventCount = 0;

typedef enum {
    SEND,
    RECV
} RequestType;

typedef struct {
    RequestType type;
    int segmented;
    int num_segments;
    MPI_Request* requests;
    MPI_Request** allocated_requests;
} SegInfo;

#define MAX_SEG_ENTRIES 10000
static struct {
    MPI_Request* key;
    SegInfo* value;
} segMap[MAX_SEG_ENTRIES];
static int segMapCount = 0;

void add_seg_info(MPI_Request* req, SegInfo* info);

SegInfo* find_seg_info(MPI_Request* req);

void remove_seg_info(MPI_Request* req);

typedef struct {
    RequestType type;
    int pipeline;

    int num_requests;
    MPI_Request* requests;

    int num_send_buffers;
    void** send_buffers;

    int source;
    void* user_buffer;
    int buffer_size;
    int tag;

    int num_segments;
    int* segment_sizes;
    MPI_Request* seg_requests;
    void** segment_buffers;
} RequestInfo;

#define MAX_REQUEST_MAP_ENTRIES 10000

static struct {
    MPI_Request* key;
    RequestInfo* value;
} requestMap[MAX_REQUEST_MAP_ENTRIES];

static int requestMapCount;

void add_request_info(MPI_Request* req, RequestInfo* info);

RequestInfo* find_request_info(MPI_Request* req);

void remove_request_info(MPI_Request* req);

static struct timespec* shared_last_time = NULL;
static MPI_Win last_time_win = MPI_WIN_NULL;
static MPI_Comm shm_comm = MPI_COMM_NULL;

static inline double get_perfect_cmp_us(double Tput, double ratio, int OriBytes, double init_sleep_us) {
    double val1 = Tput / ratio;
    double val2 = (double)OriBytes - BANDWIDTH_LIMIT * (init_sleep_us / 1e6);
    return (val1 + OriBytes - val2) / (BANDWIDTH_LIMIT + Tput) * 1e6;
}

static inline double get_perfect_cmp_us_largeTput(double Tput, double ratio, int OriBytes, double init_sleep_us) {
    double computed_sleep_us = (double)OriBytes / BANDWIDTH_LIMIT * 1e6;
    double credit_us = computed_sleep_us - init_sleep_us;
    return (Tput / (ratio * BANDWIDTH_LIMIT) * 1e6) - credit_us;
}

static inline void precise_sleep_us(double us) {
    long long sleep_time_us = (long long)(us + 1);

    struct timeval start, now;
    gettimeofday(&start, NULL);

    long long start_us = start.tv_sec * 1000000LL + start.tv_usec;
    long long target_us = start_us + sleep_time_us;

    if (sleep_time_us > 200) {
        usleep(sleep_time_us - 100);
    }

    do {
        gettimeofday(&now, NULL);
        long long now_us = now.tv_sec * 1000000LL + now.tv_usec;
        if (now_us >= target_us) break;
        sched_yield();
    } while (1);
}

static inline void precise_sleep_cycles(uint64_t cycles) {
    unsigned int aux;
    uint64_t start = __rdtsc();
    while (__rdtsc() - start < cycles);
}
static inline double timespec_to_us(const struct timespec* ts) {
    return ts->tv_sec * 1e6 + ts->tv_nsec / 1e3;
}

static inline void us_to_timespec(double us, struct timespec* ts) {
    ts->tv_sec = (time_t)(us / 1e6);
    ts->tv_nsec = (long)((us - ts->tv_sec * 1e6) * 1e3);
}

int MPI_Init(int* argc, char*** argv);

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided);

int mpi_isend_seg(const void* buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request* request);

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request* request);

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request* request);

int mpi_wait_seg(MPI_Request* request, MPI_Status* status);

int MPI_Wait(MPI_Request* request, MPI_Status* status);

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

int MPI_Send(const void* buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);

int MPI_Recv(void* buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status* status);

int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm,
                 MPI_Status* status);

int MPIR_Allgatherv_intra_ring(const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    const int * recvcounts,
    const int * displs,
    MPI_Datatype recvtype,
    MPI_Comm comm);

int MPI_Finalize(void);

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
    int blockSize);

int MPI_Allreduce_ZCCL_RI2_mt_oa_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm);

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
    int blockSize);

int MPI_Allreduce_ZCCL_RI2_st_oa_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm);

int MPI_Allreduce_ZCCL_RI2_st_oa_op_record(const void *sendbuf,
    void *recvbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm);

/**
 * Pipelined & Multidimensional data schemes
 * Aug 2025
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
    int blockSize);

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
    int blockSize);

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
    int blockSideLength);

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
    int blockSideLength);
#endif /* ----- #ifndef ZCCL_ring_H  ----- */