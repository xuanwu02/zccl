#ifndef ZCCL_appendix_H
#define ZCCL_appendix_H

#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"

size_t save_fixed_length_bytes(unsigned int *input, size_t length, unsigned char *result, int rate);

size_t extract_fixed_length_bytes(unsigned char *result, size_t length, unsigned int *input, int rate);

void ZCCL_float_openmp_threadblock_arg_2(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize);

void ZCCL_float_decompress_openmp_threadblock_arg_2(float *newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes);

int MPIR_Allgatherv_intra_ring_RI2_mt_oa_record_2(const void *sendbuf,
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

int MPIR_Allgatherv_intra_ring_RI2_mt_oa_record_seg_2(const void *sendbuf,
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

#endif