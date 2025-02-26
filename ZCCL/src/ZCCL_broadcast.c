/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */
#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"
#include "ZCCL_broadcast.h"

int MPI_Bcast_ZCCL(void *buffer,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm)
{
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint nbytes = 0;
    MPI_Status status;
    int tmp_len = 0;
    double absErrBound = compressionRatio;

    tmp_len = count;

    void *tmpbuf;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    int is_contig;
    MPI_Aint type_size;
    void *tmp_buf = NULL;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    type_size = sizeof(data_type);

    size_t outSize;
    size_t byteLength;
    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));
    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * type_size);
    if (rank == root) {
        ZCCL_float_single_thread_arg(outputBytes, buffer, &outSize, absErrBound, count, blockSize);
    }
    MPI_Bcast(&outSize, 1, MPI_INT, root, comm);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            mpi_errno =
                MPI_Recv(outputBytes, outSize, MPI_BYTE, src, MPIR_BCAST_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            break;
        }
        mask <<= 1;
    }

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            mpi_errno = MPI_Send(outputBytes, outSize, MPI_BYTE, dst, MPIR_BCAST_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
        }
        mask >>= 1;
    }

    if (rank != root) {
        ZCCL_float_decompress_single_thread_arg(
            buffer, count, absErrBound, blockSize, (char *) outputBytes);
    }
    free(outputBytes);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPI_Bcast_ZCCL_mt(void *buffer,
    float compressionRatio,
    float tolerance,
    int blockSize,
    MPI_Aint count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm)
{
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint nbytes = 0;
    MPI_Status status;
    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = count;

    void *tmpbuf;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    int is_contig;
    MPI_Aint type_size;
    void *tmp_buf = NULL;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    type_size = sizeof(data_type);

    size_t outSize;
    size_t byteLength;
    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));
    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * type_size);
    if (rank == root) {
        ZCCL_float_openmp_threadblock_arg(
            outputBytes, buffer, &outSize, absErrBound, count, blockSize);
    }
    MPI_Bcast(&outSize, 1, MPI_INT, root, comm);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            mpi_errno =
                MPI_Recv(outputBytes, outSize, MPI_BYTE, src, MPIR_BCAST_TAG, comm, &status);
            if (mpi_errno) {
                exit(-1);
            }
            break;
        }
        mask <<= 1;
    }

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            mpi_errno = MPI_Send(outputBytes, outSize, MPI_BYTE, dst, MPIR_BCAST_TAG, comm);
            if (mpi_errno) {
                exit(-1);
            }
        }
        mask >>= 1;
    }

    if (rank != root) {
        ZCCL_float_decompress_openmp_threadblock_arg(
            buffer, count, absErrBound, blockSize, (char *) outputBytes);
    }
    free(outputBytes);
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}