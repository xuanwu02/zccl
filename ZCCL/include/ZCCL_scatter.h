/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef ZCCL_BROADCAST_H
#define ZCCL_BROADCAST_H

#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"

int MPIR_Scatter_ZCCL(const void *sendbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    MPI_Aint recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm);

int MPIR_Scatter_ZCCL_mt(const void *sendbuf,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    MPI_Aint recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm);

#endif /* ----- #ifndef ZCCL_BROADCAST_H  ----- */