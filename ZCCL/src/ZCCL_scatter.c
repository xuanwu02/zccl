/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */
#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"
#include "ZCCL_scatter.h"

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
    MPI_Comm comm)
{
    MPI_Status status;
    MPI_Aint extent = 0;
    int rank, comm_size;
    int relative_rank;
    MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, src, dst;
    MPI_Aint tmp_buf_size = 0;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (rank == root)
        extent = sizeof(sendtype);

    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = sendcount * comm_size;
    void *tmpbuf;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    size_t outSize;
    size_t byteLength;
    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));
    int *compressed_offsets = (int *) malloc(comm_size * sizeof(int));
    int *compressed_lefts = (int *) malloc(comm_size * sizeof(int));

    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * sizeof(sendtype));

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    MPI_Aint nbytes;
    if (rank == root) {
        MPI_Aint sendtype_size;
        sendtype_size = sizeof(sendtype);
        nbytes = sendtype_size * sendcount;
    } else {
        MPI_Aint recvtype_size;
        recvtype_size = sizeof(recvtype);
        nbytes = recvtype_size * recvcount;
    }

    curr_cnt = 0;

    if (relative_rank && !(relative_rank % 2)) {
        tmp_buf_size = (nbytes * comm_size) / 2;
        tmp_buf = (void *) malloc(tmp_buf_size);
    }

    if (rank == root) {
        if (root != 0) {
            tmp_buf_size = nbytes * comm_size;
            tmp_buf = (void *) malloc(tmp_buf_size);

            if (recvbuf != MPI_IN_PLACE)
                memcpy(tmp_buf,
                    ((char *) sendbuf + extent * sendcount * rank),
                    sendcount * (comm_size - rank) * sizeof(sendtype));
            else
                memcpy((char *) tmp_buf + nbytes,
                    ((char *) sendbuf + extent * sendcount * (rank + 1)),
                    sendcount * (comm_size - rank - 1) * sizeof(sendtype));

            memcpy(((char *) tmp_buf + nbytes * (comm_size - rank)),
                sendbuf,
                sendcount * rank * sizeof(sendtype));

            curr_cnt = nbytes * comm_size;
        } else
            curr_cnt = sendcount * comm_size;
    }

    if (rank == root && root == 0) {
        int offset = 0;
        int i = 0, j = 0;
        for (i = 0; i < curr_cnt; i += sendcount) {
            ZCCL_float_single_thread_arg((char *) outputBytes + offset,
                (char *) sendbuf + i * sizeof(recvtype),
                &outSize,
                absErrBound,
                sendcount,
                blockSize);
            compressed_sizes[j] = outSize;
            compressed_offsets[j] = offset;
            offset += outSize;
            j++;
        }
        curr_cnt = offset;
        for (j = 0; j < comm_size; j++) {
            compressed_lefts[j] = curr_cnt - compressed_offsets[j];
        }
    } else if (rank == root && root != 0) {
        int offset = 0;
        int i = 0, j = 0;
        for (i = 0; i < curr_cnt; i += sendcount) {
            ZCCL_float_single_thread_arg((char *) outputBytes + offset,
                (char *) tmp_buf + i * sizeof(recvtype),
                &outSize,
                absErrBound,
                sendcount,
                blockSize);
            compressed_sizes[j] = outSize;
            compressed_offsets[j] = offset;
            offset += outSize;
            j++;
        }
        curr_cnt = offset;
        for (j = 0; j < comm_size; j++) {
            compressed_lefts[j] = curr_cnt - compressed_offsets[j];
        }
    }

    MPI_Bcast(compressed_sizes, comm_size, MPI_INT, root, comm);
    MPI_Bcast(compressed_offsets, comm_size, MPI_INT, root, comm);
    MPI_Bcast(compressed_lefts, comm_size, MPI_INT, root, comm);

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            if (relative_rank % 2) {
                mpi_errno = MPI_Recv(
                    outputBytes, recvcount, recvtype, src, MPIR_SCATTER_TAG, comm, &status);
                if (mpi_errno) {
                    exit(-1);
                }
            } else {
                mpi_errno = MPI_Recv(
                    outputBytes, tmp_buf_size, MPI_BYTE, src, MPIR_SCATTER_TAG, comm, &status);
                if (mpi_errno) {
                    exit(-1);
                } else
                    MPI_Get_count(&status, MPI_BYTE, &curr_cnt);
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

            if ((rank == root) && (root == 0)) {
                send_subtree_cnt = curr_cnt - compressed_offsets[mask];
                mpi_errno = MPI_Send(outputBytes + compressed_offsets[dst],
                    send_subtree_cnt,
                    MPI_BYTE,
                    dst,
                    MPIR_SCATTER_TAG,
                    comm);
            } else {
                send_subtree_cnt = compressed_offsets[dst + mask - 1] - compressed_offsets[dst]
                    + compressed_sizes[dst + mask - 1];
                mpi_errno = MPI_Send(
                    ((char *) outputBytes + compressed_offsets[dst] - compressed_offsets[rank]),
                    send_subtree_cnt,
                    MPI_BYTE,
                    dst,
                    MPIR_SCATTER_TAG,
                    comm);
            }
            if (mpi_errno) {
                exit(-1);
            }
            curr_cnt -= send_subtree_cnt;
        }
        mask >>= 1;
    }

    if ((relative_rank % 2)) {
        ZCCL_float_decompress_single_thread_arg(
            recvbuf, recvcount, absErrBound, blockSize, (char *) outputBytes);
    }
    if ((rank == root) && (root == 0) && (recvbuf != MPI_IN_PLACE)) {
        memcpy(recvbuf, sendbuf, sendcount * sizeof(sendtype));
    } else if (!(relative_rank % 2) && (recvbuf != MPI_IN_PLACE)) {
        ZCCL_float_decompress_single_thread_arg(
            recvbuf, recvcount, absErrBound, blockSize, (char *) outputBytes);
    }

    free(outputBytes);
    free(compressed_lefts);
    free(compressed_offsets);
    free(compressed_sizes);
fn_exit:
    free(tmp_buf);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

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
    MPI_Comm comm)
{
    MPI_Status status;
    MPI_Aint extent = 0;
    int rank, comm_size;
    int relative_rank;
    MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, src, dst;
    MPI_Aint tmp_buf_size = 0;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (rank == root)
        extent = sizeof(sendtype);

    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = sendcount * comm_size;
    void *tmpbuf;
    MPI_Request reqs[2];
    MPI_Status stas[2];
    size_t outSize;
    size_t byteLength;
    int *compressed_sizes = (int *) malloc(comm_size * sizeof(int));
    int *compressed_offsets = (int *) malloc(comm_size * sizeof(int));
    int *compressed_lefts = (int *) malloc(comm_size * sizeof(int));

    unsigned char *outputBytes = (unsigned char *) malloc(tmp_len * sizeof(sendtype));

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    MPI_Aint nbytes;
    if (rank == root) {
        MPI_Aint sendtype_size;
        sendtype_size = sizeof(sendtype);
        nbytes = sendtype_size * sendcount;
    } else {
        MPI_Aint recvtype_size;
        recvtype_size = sizeof(recvtype);
        nbytes = recvtype_size * recvcount;
    }

    curr_cnt = 0;

    if (relative_rank && !(relative_rank % 2)) {
        tmp_buf_size = (nbytes * comm_size) / 2;
        tmp_buf = (void *) malloc(tmp_buf_size);
    }

    if (rank == root) {
        if (root != 0) {
            tmp_buf_size = nbytes * comm_size;
            tmp_buf = (void *) malloc(tmp_buf_size);

            if (recvbuf != MPI_IN_PLACE)
                memcpy(tmp_buf,
                    ((char *) sendbuf + extent * sendcount * rank),
                    sendcount * (comm_size - rank) * sizeof(sendtype));
            else
                memcpy((char *) tmp_buf + nbytes,
                    ((char *) sendbuf + extent * sendcount * (rank + 1)),
                    sendcount * (comm_size - rank - 1) * sizeof(sendtype));

            memcpy(((char *) tmp_buf + nbytes * (comm_size - rank)),
                sendbuf,
                sendcount * rank * sizeof(sendtype));

            curr_cnt = nbytes * comm_size;
        } else
            curr_cnt = sendcount * comm_size;
    }

    if (rank == root && root == 0) {
        int offset = 0;
        int i = 0, j = 0;
        for (i = 0; i < curr_cnt; i += sendcount) {
            ZCCL_float_openmp_threadblock_arg((char *) outputBytes + offset,
                (char *) sendbuf + i * sizeof(recvtype),
                &outSize,
                absErrBound,
                sendcount,
                blockSize);
            compressed_sizes[j] = outSize;
            compressed_offsets[j] = offset;
            offset += outSize;
            j++;
        }
        curr_cnt = offset;
        for (j = 0; j < comm_size; j++) {
            compressed_lefts[j] = curr_cnt - compressed_offsets[j];
        }
    } else if (rank == root && root != 0) {
        int offset = 0;
        int i = 0, j = 0;
        for (i = 0; i < curr_cnt; i += sendcount) {
            ZCCL_float_openmp_threadblock_arg((char *) outputBytes + offset,
                (char *) tmp_buf + i * sizeof(recvtype),
                &outSize,
                absErrBound,
                sendcount,
                blockSize);
            compressed_sizes[j] = outSize;
            compressed_offsets[j] = offset;
            offset += outSize;
            j++;
        }
        curr_cnt = offset;
        for (j = 0; j < comm_size; j++) {
            compressed_lefts[j] = curr_cnt - compressed_offsets[j];
        }
    }

    MPI_Bcast(compressed_sizes, comm_size, MPI_INT, root, comm);
    MPI_Bcast(compressed_offsets, comm_size, MPI_INT, root, comm);
    MPI_Bcast(compressed_lefts, comm_size, MPI_INT, root, comm);

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            if (relative_rank % 2) {
                mpi_errno = MPI_Recv(
                    outputBytes, recvcount, recvtype, src, MPIR_SCATTER_TAG, comm, &status);
                if (mpi_errno) {
                    exit(-1);
                }
            } else {
                mpi_errno = MPI_Recv(
                    outputBytes, tmp_buf_size, MPI_BYTE, src, MPIR_SCATTER_TAG, comm, &status);
                if (mpi_errno) {
                    exit(-1);
                } else
                    MPI_Get_count(&status, MPI_BYTE, &curr_cnt);
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

            if ((rank == root) && (root == 0)) {
                send_subtree_cnt = curr_cnt - compressed_offsets[mask];
                mpi_errno = MPI_Send(outputBytes + compressed_offsets[dst],
                    send_subtree_cnt,
                    MPI_BYTE,
                    dst,
                    MPIR_SCATTER_TAG,
                    comm);
            } else {
                send_subtree_cnt = compressed_offsets[dst + mask - 1] - compressed_offsets[dst]
                    + compressed_sizes[dst + mask - 1];
                mpi_errno = MPI_Send(
                    ((char *) outputBytes + compressed_offsets[dst] - compressed_offsets[rank]),
                    send_subtree_cnt,
                    MPI_BYTE,
                    dst,
                    MPIR_SCATTER_TAG,
                    comm);
            }
            if (mpi_errno) {
                exit(-1);
            }
            curr_cnt -= send_subtree_cnt;
        }
        mask >>= 1;
    }

    if ((relative_rank % 2)) {
        ZCCL_float_decompress_openmp_threadblock_arg(
            recvbuf, recvcount, absErrBound, blockSize, (char *) outputBytes);
    }
    if ((rank == root) && (root == 0) && (recvbuf != MPI_IN_PLACE)) {
        memcpy(recvbuf, sendbuf, sendcount * sizeof(sendtype));
    } else if (!(relative_rank % 2) && (recvbuf != MPI_IN_PLACE)) {
        ZCCL_float_decompress_openmp_threadblock_arg(
            recvbuf, recvcount, absErrBound, blockSize, (char *) outputBytes);
    }

    free(outputBytes);
    free(compressed_lefts);
    free(compressed_offsets);
    free(compressed_sizes);
fn_exit:
    free(tmp_buf);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
