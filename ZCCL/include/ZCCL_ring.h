/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef ZCCL_ring_H
#define ZCCL_ring_H

#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"

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

#endif /* ----- #ifndef ZCCL_ring_H  ----- */