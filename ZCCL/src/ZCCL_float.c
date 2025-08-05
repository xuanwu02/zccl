/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "ZCCL.h"
#include "ZCCL_float.h"
#include <assert.h>
#include <math.h>
#include "ZCCL_TypeManager.h"
#include "ZCCL_BytesToolkit.h"

#ifdef _OPENMP
#include "omp.h"
#endif

unsigned char *ZCCL_float_openmp_direct_predict_quantization(float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t i = 0;

    int *quti_arr = (int *) malloc(nbEle * sizeof(int));
    int *diff_arr = (int *) malloc(nbEle * sizeof(int));
    (*outSize) = 0;

    int nbThreads = 1;
    double inver_bound = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();

            inver_bound = 1 / absErrBound;
        }

#pragma omp for schedule(static)
        for (i = 0; i < nbEle; i++) {
            quti_arr[i] = (op[i] + absErrBound) * inver_bound;
        }

#pragma omp single
        {
            diff_arr[0] = quti_arr[0];
        }

#pragma omp for schedule(static)
        for (i = 1; i < nbEle; i++) {
            diff_arr[i] = quti_arr[i] - quti_arr[i - 1];
        }
    }

    free(quti_arr);

    return diff_arr;
#else
    return NULL;
#endif
}

unsigned char *ZCCL_float_openmp_threadblock_predict_quantization(float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    int *diff_arr = (int *) malloc(nbEle * sizeof(int));
    (*outSize) = 0;

    int nbThreads = 1;
    double inver_bound = 1;
    int threadblocksize = 1;
    int remainder = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();

            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
        }
        size_t i = 0;

        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;
        int prior = 0;
        int current = 0;
        prior = (op[lo] + absErrBound) * inver_bound;
        diff_arr[lo] = prior;
        for (i = lo + 1; i < hi; i++) {
            current = (op[i] + absErrBound) * inver_bound;
            diff_arr[i] = current - prior;
            prior = current;
        }
#pragma omp single
        {
            if (remainder != 0) {
                size_t remainder_lo = nbEle - remainder;
                prior = (op[remainder_lo] + absErrBound) * inver_bound;
                diff_arr[remainder_lo] = prior;
                for (i = nbEle - remainder + 1; i < nbEle; i++) {
                    current = (op[i] + absErrBound) * inver_bound;
                    diff_arr[i] = current - prior;
                    prior = current;
                }
            }
        }
    }

    return diff_arr;
#else
    return NULL;
#endif
}

unsigned char *ZCCL_fast_compress_args(int fastMode,
    int dataType,
    void *data,
    size_t *outSize,
    int errBoundMode,
    float absErrBound,
    float relBoundRatio,
    size_t r5,
    size_t r4,
    size_t r3,
    size_t r2,
    size_t r1)
{
    unsigned char *bytes = NULL;
    size_t length = computeDataLength(r5, r4, r3, r2, r1);
    size_t i = 0;
    int blockSize = 128;
    if (dataType == SZ_FLOAT) {
        float realPrecision = absErrBound;
        if (errBoundMode == REL) {
            float *oriData = (float *) data;
            float min = oriData[0];
            float max = oriData[0];
            for (i = 0; i < length; i++) {
                float v = oriData[i];
                if (min > v)
                    min = v;
                else if (max < v)
                    max = v;
            }
            float valueRange = max - min;
            realPrecision = valueRange * relBoundRatio;
            printf("REAL ERROR BOUND IS %20f\n", realPrecision);
            if (fastMode == 1) {
                bytes = ZCCL_float_openmp_threadblock(
                    oriData, outSize, realPrecision, length, blockSize);
            } else if (fastMode == 2) {
                bytes = ZCCL_float_openmp_threadblock_randomaccess(
                    oriData, outSize, realPrecision, length, blockSize);
            }
        }
        if (fastMode == 1) {
            bytes = ZCCL_float_openmp_threadblock(
                (float *) data, outSize, realPrecision, length, blockSize);
        } else if (fastMode == 2) {
            bytes = ZCCL_float_openmp_threadblock_randomaccess(
                (float *) data, outSize, realPrecision, length, blockSize);
        }
    } else {
        printf("ZCCL only supports float type for now\n");
    }
    return bytes;
}

unsigned char *ZCCL_float_openmp_threadblock(float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
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

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
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

    return outputBytes;
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

void ZCCL_float_openmp_threadblock_arg(unsigned char *outputBytes,
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

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                            temp_sign_arr, num_remainder_in_rm, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, num_remainder_in_rm, block_pointer, bit_count);
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

void ZCCL_float_single_thread_arg(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

    int nbThreads = 1;
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

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    unsigned char *outputBytes_perthread =
        (unsigned char *) malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;

    int tid = 0;
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
                signbytelength =
                    convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;

                savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, num_remainder_in_tb, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }
    }

    outSize_perthread_arr[tid] = outSize_perthread;

    offsets_perthread_arr[0] = 0;

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));

    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);

    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);

    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
}

size_t ZCCL_float_single_thread_arg_record(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
    size_t total_memaccess = 0;

    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;
    total_memaccess += sizeof(size_t);

    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

    int nbThreads = 1;
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

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    unsigned char *outputBytes_perthread =
        (unsigned char *) malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;

    int tid = 0;
    int lo = tid * threadblocksize;
    int hi = (tid + 1) * threadblocksize;

    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;
    prior = (op[lo]) * inver_bound;
    total_memaccess += sizeof(float);

    memcpy(block_pointer, &prior, sizeof(int));
    total_memaccess += sizeof(int);
    total_memaccess += sizeof(int);

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
                total_memaccess += sizeof(float);
                diff = current - prior;
                prior = current;
                if (diff == 0) {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                    total_memaccess += sizeof(unsigned int);
                    total_memaccess += sizeof(unsigned int);
                } else if (diff > 0) {
                    temp_sign_arr[j] = 0;
                    total_memaccess += sizeof(unsigned int);
                    if (diff > max) {
                        max = diff;
                    }
                    temp_predict_arr[j] = diff;
                    total_memaccess += sizeof(unsigned int);
                } else if (diff < 0) {
                    temp_sign_arr[j] = 1;
                    total_memaccess += sizeof(unsigned int);
                    diff = 0 - diff;
                    if (diff > max) {
                        max = diff;
                    }
                    temp_predict_arr[j] = diff;
                    total_memaccess += sizeof(unsigned int);
                }
            }
            if (max == 0) {
                block_pointer[0] = 0;
                total_memaccess += sizeof(unsigned char);
                block_pointer++;
                outSize_perthread++;
            } else {
                bit_count = (int) (log2f(max)) + 1;
                block_pointer[0] = bit_count;
                total_memaccess += sizeof(unsigned char);

                outSize_perthread++;
                block_pointer++;
                signbytelength =
                    convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer);
                total_memaccess += (sizeof(unsigned int) * blockSize);
                block_pointer += signbytelength;
                total_memaccess += (sizeof(unsigned char) * signbytelength);
                outSize_perthread += signbytelength;

                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, blockSize, block_pointer, bit_count);
                total_memaccess += (sizeof(unsigned int) * blockSize);

                block_pointer += savedbitsbytelength;
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
                outSize_perthread += savedbitsbytelength;
            }
        }
    }

    if (num_remainder_in_tb > 0) {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
            max = 0;
            for (j = 0; j < num_remainder_in_tb; j++) {
                current = (op[i + j]) * inver_bound;
                total_memaccess += sizeof(float);
                diff = current - prior;
                prior = current;
                if (diff == 0) {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                    total_memaccess += sizeof(unsigned int);
                    total_memaccess += sizeof(unsigned int);
                } else if (diff > 0) {
                    temp_sign_arr[j] = 0;
                    total_memaccess += sizeof(unsigned int);
                    if (diff > max) {
                        max = diff;
                    }
                    temp_predict_arr[j] = diff;
                    total_memaccess += sizeof(unsigned int);
                } else if (diff < 0) {
                    temp_sign_arr[j] = 1;
                    total_memaccess += sizeof(unsigned int);
                    diff = 0 - diff;
                    if (diff > max) {
                        max = diff;
                    }
                    temp_predict_arr[j] = diff;
                    total_memaccess += sizeof(unsigned int);
                }
            }
            if (max == 0) {
                block_pointer[0] = 0;
                total_memaccess += sizeof(unsigned char);
                block_pointer++;
                outSize_perthread++;
            } else {
                bit_count = (int) (log2f(max)) + 1;
                block_pointer[0] = bit_count;
                total_memaccess += sizeof(unsigned char);

                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(
                    temp_sign_arr, num_remainder_in_tb, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * signbytelength);
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, num_remainder_in_tb, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
            }
        }
    }

    outSize_perthread_arr[tid] = outSize_perthread;
    total_memaccess += sizeof(size_t);

    offsets_perthread_arr[0] = 0;
    total_memaccess += sizeof(size_t);

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    total_memaccess += (sizeof(size_t) * 3);
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
    total_memaccess += (sizeof(unsigned char) * nbThreads * sizeof(size_t));
    total_memaccess += (sizeof(unsigned char) * nbThreads * sizeof(size_t));

    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
    total_memaccess += (sizeof(unsigned char) * outSize_perthread);
    total_memaccess += (sizeof(unsigned char) * outSize_perthread);

    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);

    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
    return total_memaccess;
}

unsigned char *ZCCL_float_openmp_threadblock_randomaccess(float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int new_block_size = block_size - 1;
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
            num_full_block_in_tb = (threadblocksize) / block_size;
            num_remainder_in_tb = (threadblocksize) % block_size;
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

        unsigned char *temp_sign_arr =
            (unsigned char *) malloc(new_block_size * sizeof(unsigned char));

        unsigned int *temp_predict_arr =
            (unsigned int *) malloc(new_block_size * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;

        if (num_full_block_in_tb > 0) {
            for (i = lo; i < hi - num_remainder_in_tb; i = i + block_size) {
                max = 0;
                prior = (op[i]) * inver_bound;
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(unsigned int);
                outSize_perthread += sizeof(unsigned int);
                for (j = 0; j < new_block_size; j++) {
                    current = (op[i + j + 1]) * inver_bound;
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
                        temp_sign_arr, new_block_size, block_pointer);
                    block_pointer += signbytelength;
                    outSize_perthread += signbytelength;

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
                        temp_predict_arr, new_block_size, block_pointer, bit_count);

                    block_pointer += savedbitsbytelength;
                    outSize_perthread += savedbitsbytelength;
                }
            }
        }

        if (num_remainder_in_tb > 0) {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
                prior = (op[i]) * inver_bound;
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(unsigned int);
                outSize_perthread += sizeof(unsigned int);
                max = 0;
                for (j = 0; j < num_remainder_in_tb - 1; j++) {
                    current = (op[i + j + 1]) * inver_bound;
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
                        temp_sign_arr, num_remainder_in_tb - 1, block_pointer);
                    block_pointer += signbytelength;
                    outSize_perthread += signbytelength;
                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
                        temp_predict_arr, num_remainder_in_tb - 1, block_pointer, bit_count);
                    block_pointer += savedbitsbytelength;
                    outSize_perthread += savedbitsbytelength;
                }
            }
        }

        if (tid == nbThreads - 1 && remainder != 0) {
            unsigned int num_full_block_in_rm = (remainder) / block_size;
            unsigned int num_remainder_in_rm = (remainder) % block_size;

            if (num_full_block_in_rm > 0) {
                for (i = hi; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    prior = (op[i]) * inver_bound;
                    memcpy(block_pointer, &prior, sizeof(int));
                    block_pointer += sizeof(unsigned int);
                    outSize_perthread += sizeof(unsigned int);
                    max = 0;
                    for (j = 0; j < new_block_size; j++) {
                        current = (op[i + j + 1]) * inver_bound;
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
                            temp_sign_arr, new_block_size, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, new_block_size, block_pointer, bit_count);
                        block_pointer += savedbitsbytelength;
                        outSize_perthread += savedbitsbytelength;
                    }
                }
            }
            if (num_remainder_in_rm > 0) {
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size) {
                    max = 0;
                    prior = (op[i]) * inver_bound;
                    memcpy(block_pointer, &prior, sizeof(int));
                    block_pointer += sizeof(unsigned int);
                    outSize_perthread += sizeof(unsigned int);
                    for (j = 0; j < num_remainder_in_rm - 1; j++) {
                        current = (op[i + j + 1]) * inver_bound;
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
                            temp_sign_arr, num_remainder_in_tb - 1, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, num_remainder_in_tb - 1, block_pointer, bit_count);
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
#pragma omp barrier

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    return outputBytes;
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

void ZCCL_float_single_thread_arg_split_record(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize,
    unsigned char *chunk_arr,
    size_t chunk_iter)
{
    float *op = oriData;

    size_t *arr = (size_t *) chunk_arr;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

    int nbThreads = 1;
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

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    unsigned char *outputBytes_perthread =
        (unsigned char *) malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    int tid = 0;
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
                signbytelength =
                    convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
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
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, num_remainder_in_tb, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }
    }

    outSize_perthread_arr[tid] = outSize_perthread;

    offsets_perthread_arr[0] = 0;

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));

    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);

    size_t final_size = (*outSize);
    arr[chunk_iter] = final_size;

    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
}

void SZp_compress3D_fast(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_3d size;
    DSize3D_init(&size, dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    size_t block_offset_0 = size.Bsize * size.Bsize;
    int * col_buffer = (int *)calloc(block_offset_0, sizeof(int));
    int * prevSlice_buffer = (int *)calloc(size.offset_0, sizeof(int));
    int ** prevRow_buffer = (int **)malloc(size.Bsize*sizeof(int *));
    for(int i=0; i<size.Bsize; i++) prevRow_buffer[i] = (int *)calloc(size.dim3, sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const float * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        const float * y_data_pos = x_data_pos;
        for(int i=0; i<size.Bsize; i++) memset(prevRow_buffer[i], 0, size.dim3*sizeof(int));
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            const float * z_data_pos = y_data_pos;
            memset(col_buffer, 0, block_offset_0*sizeof(int));
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                unsigned int * abs_err_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                const float * curr_data_pos = z_data_pos;
                int offset = y * size.Bsize * size.dim3 + z * size.Bsize;
                int max_err = 0;
                for(int i=0; i<size_x; i++){
                    int * prevRow = prevRow_buffer[i] + z * size.Bsize;
                    int * prevSlice = prevSlice_buffer + offset;
                    for(int j=0; j<size_y; j++){
                        int index = i*size.Bsize+j;
                        int prevLeft = col_buffer[index];
                        int * prevSlice_pos = prevSlice + j * size.offset_1;
                        for(int k=0; k<size_z; k++){
                            int q = SZ_quantize(*curr_data_pos, inver_eb);
                            int err_1 = q - prevLeft;
                            int err_2 = err_1 - prevRow[k];
                            int err_3 = err_2 - prevSlice_pos[k];
                            prevRow[k] = err_1;
                            prevSlice_pos[k] = err_2;
                            prevLeft = q;
                            (*sign_pos++) = (err_3 < 0);
                            unsigned int u = abs(err_3);
                            (*abs_err_pos++) = u;
                            max_err = max_err > u ? max_err : u;
                            curr_data_pos++;
                        }
                        col_buffer[index] = prevLeft;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                z_data_pos += size_z;
                int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                    cmpData_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, cmpData_pos, fixed_rate);
                    cmpData_pos += savedbitsbyteLength;
                }
            }
            y_data_pos += size.Bsize * size.offset_1;
        }
        x_data_pos += size.Bsize * size.offset_0;
    }
    *cmpSize = cmpData_pos - cmpData;
    for(int i=0; i<size.Bsize; i++) free(prevRow_buffer[i]);
    free(col_buffer);
    free(prevSlice_buffer);
    free(prevRow_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_compress3D(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_3d size;
    DSize3D_init(&size, dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const float * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const float * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            const float * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                unsigned int * abs_diff_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                int quant_diff, max_quant_diff = 0;
                const float * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            quant_diff = predict_lorenzo_3d(curr_data_pos++, curr_buffer_pos++, inver_eb, offset_0, offset_1);
                            (*sign_pos++) = (quant_diff < 0);
                            unsigned int abs_diff = abs(quant_diff);
                            (*abs_diff_pos++) = abs_diff;
                            max_quant_diff = max_quant_diff > abs_diff ? max_quant_diff : abs_diff;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
                int fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                    cmpData_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, cmpData_pos, fixed_rate);
                    cmpData_pos += savedbitsbyteLength;
                }
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
    *cmpSize = cmpData_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_compress3D_fast_openmp(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize
){
#ifdef _OPENMP
    float *op = oriData;
    double inver_eb = 0.5 / errorBound;

    size_t nbEle = dim1 * dim2 * dim3;
    size_t gloff_0 = dim2 * dim3;
    size_t gloff_1 = dim3;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*cmpSize) = 0;

    size_t base_x, base_y, base_z;
    size_t rem_x, rem_y, rem_z;

    Meta *meta = NULL;
    size_t gloffset;

#pragma omp parallel
    {
        int nbThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
#pragma omp single
        {
            real_outputBytes = cmpData + nbThreads * sizeof(size_t);
            (*cmpSize) += nbThreads * sizeof(size_t);

            outSize_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;

            int tdims[3] = {0,0,0};
            My_Dims_create(nbThreads, 3, tdims);
            base_x = dim1 / tdims[0], rem_x = dim1 % tdims[0];
            base_y = dim2 / tdims[1], rem_y = dim2 % tdims[1];
            base_z = dim3 / tdims[2], rem_z = dim3 % tdims[2];
            // meta
            meta = (Meta*)malloc(nbThreads * sizeof(Meta));
            int ox_acc = 0, r = 0;
            for (int px = 0; px < tdims[0]; px++) {
                int sx = base_x + (px < rem_x ? 1 : 0);
                int ox = ox_acc;
                ox_acc += sx;

                int oy_acc = 0;
                for (int py = 0; py < tdims[1]; py++) {
                    int sy = base_y + (py < rem_y ? 1 : 0);
                    int oy = oy_acc;
                    oy_acc += sy;

                    int oz_acc = 0;
                    for (int pz = 0; pz < tdims[2]; pz++) {
                        int sz = base_z + (pz < rem_z ? 1 : 0);
                        int oz = oz_acc;
                        oz_acc += sz;

                        int ls = sz * sy * sz;
                        meta[r++] = (Meta){ox, oy, oz, sx, sy, sz, ls};
                    }
                }
            }
        }
#pragma omp barrier
        // printf("cmp -> tid = %d / %d\n", tid,nbThreads);
        // fflush(stdout);
#pragma omp barrier

        Meta m = meta[tid];
        gloffset = m.ox * gloff_0 + m.oy * gloff_1 + m.oz;

        DSize_3d size;
        DSize3D_init(&size, m.sx, m.sy, m.sz, blockSideLength);

        unsigned char *outputBytes_perthread = (unsigned char *) malloc(maxPreservedBufferSize_perthread);
        unsigned char * thr_pos = outputBytes_perthread + size.num_blocks * FIXED_RATE_PER_BLOCK_BYTES;

        size_t block_offset_0 = size.Bsize * size.Bsize;
        int * col_buffer = (int *)calloc(block_offset_0, sizeof(int));
        int * prevSlice_buffer = (int *)calloc(size.offset_0, sizeof(int));
        int ** prevRow_buffer = (int **)malloc(size.Bsize*sizeof(int *));
        for(int i=0; i<size.Bsize; i++) prevRow_buffer[i] = (int *)calloc(size.dim3, sizeof(int));
        unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
        unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
        const float * x_data_pos = op + gloffset;
        int block_ind = 0;
        for(size_t x=0; x<size.block_dim1; x++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            const float * y_data_pos = x_data_pos;
            for(int i=0; i<size.Bsize; i++) memset(prevRow_buffer[i], 0, size.dim3*sizeof(int));
            for(size_t y=0; y<size.block_dim2; y++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                const float * z_data_pos = y_data_pos;
                memset(col_buffer, 0, block_offset_0*sizeof(int));
                for(size_t z=0; z<size.block_dim3; z++){
                    int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                    int block_size = size_x * size_y * size_z;
                    unsigned int * abs_err_pos = absPredError;
                    unsigned char * sign_pos = signFlag;
                    const float * curr_data_pos = z_data_pos;
                    int offset = y * size.Bsize * size.dim3 + z * size.Bsize;
                    int max_err = 0;
                    for(int i=0; i<size_x; i++){
                        int * prevRow = prevRow_buffer[i] + z * size.Bsize;
                        int * prevSlice = prevSlice_buffer + offset;
                        for(int j=0; j<size_y; j++){
                            int index = i*size.Bsize+j;
                            int prevLeft = col_buffer[index];
                            int * prevSlice_pos = prevSlice + j * size.offset_1;
                            for(int k=0; k<size_z; k++){
                                int q = SZ_quantize(*curr_data_pos, inver_eb);
                                int err_1 = q - prevLeft;
                                int err_2 = err_1 - prevRow[k];
                                int err_3 = err_2 - prevSlice_pos[k];
                                prevRow[k] = err_1;
                                prevSlice_pos[k] = err_2;
                                prevLeft = q;
                                (*sign_pos++) = (err_3 < 0);
                                unsigned int u = abs(err_3);
                                (*abs_err_pos++) = u;
                                max_err = max_err > u ? max_err : u;
                                curr_data_pos++;
                            }
                            col_buffer[index] = prevLeft;
                            curr_data_pos += gloff_1 - size_z;
                        }
                        curr_data_pos += gloff_0 - size_y * gloff_1;
                    }
                    z_data_pos += size_z;
                    int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
                    outputBytes_perthread[block_ind++] = (unsigned char)fixed_rate;
                    if(fixed_rate){
                        unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, thr_pos);
                        thr_pos += signbyteLength;
                        unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, thr_pos, fixed_rate);
                        thr_pos += savedbitsbyteLength;
                    }
                }
                y_data_pos += size.Bsize * gloff_1;
            }
            x_data_pos += size.Bsize * gloff_0;
        }

        size_t outSize_perthread = thr_pos - outputBytes_perthread;
        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (int i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*cmpSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
            memcpy(cmpData, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier

        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);

        free(col_buffer);
        free(prevSlice_buffer);
        free(prevRow_buffer);
        free(outputBytes_perthread);
        free(signFlag);
        free(absPredError);
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
