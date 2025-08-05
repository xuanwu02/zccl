#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "mpi.h"
// #include "ZCCL.h"
#include "ZCCL_TypeManager.h"
#include "ZCCL_BytesToolkit.h"
#include "ZCCL_appendix.h"
#include "ZCCL_multidim.h"

#ifdef _OPENMP
#include "omp.h"
#endif

size_t save_fixed_length_bytes(unsigned int *input, size_t length, unsigned char *result, int rate) {
    int byte_count;

    if (rate < 8)
        byte_count = 1;
    else if (rate < 16)
        byte_count = 2;
    else if (rate < 24)
        byte_count = 3;
    else
        byte_count = 4;

    for (size_t i = 0; i < length; i++) {
        unsigned int val = input[i];
        for (int b = 0; b < byte_count; b++) {
            result[i * byte_count + b] = (val >> (8 * b)) & 0xFF;
        }
    }

    size_t byteLength = length * byte_count;
    return byteLength;
}

size_t extract_fixed_length_bytes(unsigned char *result, size_t length, unsigned int *input, int rate) {
    int byte_count;

    if (rate < 8)
        byte_count = 1;
    else if (rate < 16)
        byte_count = 2;
    else if (rate < 24)
        byte_count = 3;
    else
        byte_count = 4;

    for (size_t i = 0; i < length; i++) {
        unsigned int val = 0;
        for (int b = 0; b < byte_count; b++) {
            val |= ((unsigned int)result[i * byte_count + b]) << (8 * b);
        }
        input[i] = val;
    }

    size_t byteLength = length * byte_count;
    return byteLength;
}

void ZCCL_float_openmp_threadblock_arg_2(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));

            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            num_full_block_in_tb = (threadblocksize - 1) / block_size;
            num_remainder_in_tb = (threadblocksize - 1) % block_size;
        }
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;
        unsigned char *outputBytes_perthread =
            (unsigned char *) malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;

        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        prior = (op[lo]) * inver_bound;

        memcpy(block_pointer, &prior, sizeof(int));

        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);

        unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;

        if (num_full_block_in_tb > 0) {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
                max = 0;
                for (j = 0; j < block_size; j++) {
                    current = (op[i + j]) * inver_bound;
                    diff = current - prior;
                    prior = current;
                    if (diff == 0) {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    } else if (diff > 0) {
                        temp_sign_arr[j] = 0;
                        if (diff > max) {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    } else if (diff < 0) {
                        temp_sign_arr[j] = 1;
                        diff = 0 - diff;
                        if (diff > max) {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                }
                if (max == 0) {
                    block_pointer[0] = 0;
                    block_pointer++;
                    outSize_perthread++;
                } else {
                    bit_count = (int) (log2f(max)) + 1;
                    block_pointer[0] = bit_count;

                    outSize_perthread++;
                    block_pointer++;
                    signbytelength = convertIntArray2ByteArray_fast_1b_args(
                        temp_sign_arr, blockSize, block_pointer);
                    block_pointer += signbytelength;
                    outSize_perthread += signbytelength;

                    savedbitsbytelength = save_fixed_length_bytes(
                        temp_predict_arr, blockSize, block_pointer, bit_count);

                    block_pointer += savedbitsbytelength;
                    outSize_perthread += savedbitsbytelength;
                }
            }
        }

        if (num_remainder_in_tb > 0) {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
                max = 0;
                for (j = 0; j < num_remainder_in_tb; j++) {
                    current = (op[i + j]) * inver_bound;
                    diff = current - prior;
                    prior = current;
                    if (diff == 0) {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    } else if (diff > 0) {
                        temp_sign_arr[j] = 0;
                        if (diff > max) {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    } else if (diff < 0) {
                        temp_sign_arr[j] = 1;
                        diff = 0 - diff;
                        if (diff > max) {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                }
                if (max == 0) {
                    block_pointer[0] = 0;
                    block_pointer++;
                    outSize_perthread++;
                } else {
                    bit_count = (int) (log2f(max)) + 1;
                    block_pointer[0] = bit_count;

                    outSize_perthread++;
                    block_pointer++;
                    signbytelength = convertIntArray2ByteArray_fast_1b_args(
                        temp_sign_arr, num_remainder_in_tb, block_pointer);
                    block_pointer += signbytelength;
                    outSize_perthread += signbytelength;
                    savedbitsbytelength = save_fixed_length_bytes(
                        temp_predict_arr, num_remainder_in_tb, block_pointer, bit_count);
                    block_pointer += savedbitsbytelength;
                    outSize_perthread += savedbitsbytelength;
                }
            }
        }

        if (tid == nbThreads - 1 && remainder != 0) {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size;
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            prior = (op[hi]) * inver_bound;

            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(int);
            outSize_perthread += sizeof(int);
            if (num_full_block_in_rm > 0) {
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    max = 0;
                    for (j = 0; j < block_size; j++) {
                        current = (op[i + j]) * inver_bound;
                        diff = current - prior;
                        prior = current;
                        if (diff == 0) {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        } else if (diff > 0) {
                            temp_sign_arr[j] = 0;
                            if (diff > max) {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        } else if (diff < 0) {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max) {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) {
                        block_pointer[0] = 0;
                        block_pointer++;
                        outSize_perthread++;
                    } else {
                        bit_count = (int) (log2f(max)) + 1;
                        block_pointer[0] = bit_count;

                        outSize_perthread++;
                        block_pointer++;
                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, blockSize, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = save_fixed_length_bytes(
                            temp_predict_arr, blockSize, block_pointer, bit_count);
                        block_pointer += savedbitsbytelength;
                        outSize_perthread += savedbitsbytelength;
                    }
                }
            }
            if (num_remainder_in_rm > 0) {
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size) {
                    max = 0;
                    for (j = 0; j < num_remainder_in_rm; j++) {
                        current = (op[i + j]) * inver_bound;
                        diff = current - prior;
                        prior = current;
                        if (diff == 0) {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        } else if (diff > 0) {
                            temp_sign_arr[j] = 0;
                            if (diff > max) {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        } else if (diff < 0) {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max) {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) {
                        block_pointer[0] = 0;
                        block_pointer++;
                        outSize_perthread++;
                    } else {
                        bit_count = (int) (log2f(max)) + 1;
                        block_pointer[0] = bit_count;

                        outSize_perthread++;
                        block_pointer++;
                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, num_remainder_in_tb, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = save_fixed_length_bytes(
                            temp_predict_arr, num_remainder_in_tb, block_pointer, bit_count);
                        block_pointer += savedbitsbytelength;
                        outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] =
                    offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) +=
                offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid],
            outputBytes_perthread,
            outSize_perthread);

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void ZCCL_float_decompress_openmp_threadblock_arg_2(float *newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes)
{
#ifdef _OPENMP

    size_t *offsets = (size_t *) cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    int threadblocksize = 0;
    int remainder = 0;
    int block_size = blockSize;
    int num_full_block_in_tb = 0;
    int num_remainder_in_tb = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            num_full_block_in_tb = (threadblocksize - 1) / block_size;
            num_remainder_in_tb = (threadblocksize - 1) % block_size;
        }
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;
        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        int max = 0;
        int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid];
        unsigned char *block_pointer = outputBytes_perthread;
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        float ori_prior = 0.0;
        float ori_current = 0.0;

        ori_prior = (float) prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float));
        newData_perthread += 1;

        unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;
        if (num_full_block_in_tb > 0) {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0) {
                    ori_prior = (float) prior * absErrBound;

                    for (j = 0; j < block_size; j++) {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                } else {
                    convertByteArray2IntArray_fast_1b_args(
                        block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = extract_fixed_length_bytes(
                        block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < block_size; j++) {
                        if (temp_sign_arr[j] == 0) {
                            diff = temp_predict_arr[j];
                        } else {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float) current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }

        if (num_remainder_in_tb > 0) {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count == 0) {
                    ori_prior = (float) prior * absErrBound;
                    for (j = 0; j < num_remainder_in_tb; j++) {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                } else {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                        block_pointer,
                        (num_remainder_in_tb - 1) / 8 + 1,
                        temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = extract_fixed_length_bytes(
                        block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < num_remainder_in_tb; j++) {
                        if (temp_sign_arr[j] == 0) {
                            diff = temp_predict_arr[j];
                        } else {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float) current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }

        if (tid == nbThreads - 1 && remainder != 0) {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size;
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            ori_prior = (float) prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
            if (num_full_block_in_rm > 0) {
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0) {
                        ori_prior = (float) prior * absErrBound;
                        for (j = 0; j < block_size; j++) {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    } else {
                        convertByteArray2IntArray_fast_1b_args(
                            block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = extract_fixed_length_bytes(
                            block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < block_size; j++) {
                            if (temp_sign_arr[j] == 0) {
                                diff = temp_predict_arr[j];
                            } else {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float) current * absErrBound;
                            prior = current;
                            memcpy(newData_perthread, &ori_current, sizeof(float));
                            newData_perthread++;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0) {
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0) {
                        ori_prior = (float) prior * absErrBound;
                        for (j = 0; j < num_remainder_in_rm; j++) {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    } else {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm,
                            block_pointer,
                            (num_remainder_in_rm - 1) / 8 + 1,
                            temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = extract_fixed_length_bytes(
                            block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < num_remainder_in_rm; j++) {
                            if (temp_sign_arr[j] == 0) {
                                diff = temp_predict_arr[j];
                            } else {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float) current * absErrBound;
                            prior = current;
                            memcpy(newData_perthread, &ori_current, sizeof(float));
                            newData_perthread++;
                        }
                    }
                }
            }
        }
#pragma omp barrier
        free(temp_predict_arr);
        free(temp_sign_arr);
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

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

    ZCCL_float_openmp_threadblock_arg_2(
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
        ZCCL_float_decompress_openmp_threadblock_arg_2((char *) recvbuf + displs[i] * recvtype_extent,
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
        ZCCL_float_openmp_threadblock_arg_2(bytes[i] + sizeof(size_t), (unsigned char *) recvbuf + displs[rank] * recvtype_extent + send_seg_displs[i], &outSize, absErrBound, send_seg_cnts[i] / recvtype_extent, blockSize);
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
                        ZCCL_float_decompress_openmp_threadblock_arg_2((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
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
                    ZCCL_float_decompress_openmp_threadblock_arg_2((unsigned char *) recvbuf + displs[ridx] * recvtype_extent + recv_seg_displs[j],
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
