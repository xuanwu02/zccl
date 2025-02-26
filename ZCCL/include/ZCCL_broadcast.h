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

int MPI_Bcast_ZCCL(void *buffer,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm);

int MPI_Bcast_ZCCL_mt(void *buffer,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm);

#endif /* ----- #ifndef ZCCL_BROADCAST_H  ----- */
