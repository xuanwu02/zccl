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
#include "ZCCLd_float.h"
#include <assert.h>
#include <math.h>
#include "ZCCL_TypeManager.h"
#include "ZCCL_BytesToolkit.h"

#ifdef _OPENMP
#include "omp.h"
#endif

void ZCCL_float_decompress_openmp_threadblock(float **newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes)
{
#ifdef _OPENMP
    *newData = (float *) malloc(sizeof(float) * nbEle);
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
        float *newData_perthread = *newData + lo;
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
                if (bit_count >= 32) {
                    printf(
                        "In decompression: num_full_block_in_tb i %zu, bit_count %u at thread %d\n",
                        i,
                        bit_count,
                        tid);
                }

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

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

void ZCCL_float_decompress_single_thread_arg(float *newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes)
{
    size_t *offsets = (size_t *) cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    int threadblocksize = 0;

    int block_size = blockSize;
    int num_full_block_in_tb = 0;
    int num_remainder_in_tb = 0;

    nbThreads = 1;
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

    num_full_block_in_tb = (threadblocksize - 1) / block_size;
    num_remainder_in_tb = (threadblocksize - 1) % block_size;

    int tid = 0;
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

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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
    free(temp_predict_arr);
    free(temp_sign_arr);
}

size_t ZCCL_float_decompress_single_thread_arg_record(float *newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes)
{
    size_t total_memaccess = 0;

    size_t *offsets = (size_t *) cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    int threadblocksize = 0;

    int block_size = blockSize;
    int num_full_block_in_tb = 0;
    int num_remainder_in_tb = 0;

    nbThreads = 1;
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

    num_full_block_in_tb = (threadblocksize - 1) / block_size;
    num_remainder_in_tb = (threadblocksize - 1) % block_size;

    int tid = 0;
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
    total_memaccess += sizeof(size_t);
    unsigned char *block_pointer = outputBytes_perthread;
    memcpy(&prior, block_pointer, sizeof(int));
    block_pointer += sizeof(unsigned int);

    total_memaccess += sizeof(int);
    total_memaccess += sizeof(int);

    float ori_prior = 0.0;
    float ori_current = 0.0;

    ori_prior = (float) prior * absErrBound;
    memcpy(newData_perthread, &ori_prior, sizeof(float));
    total_memaccess += sizeof(float);
    total_memaccess += sizeof(float);
    newData_perthread += 1;

    unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

    unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0;
    unsigned int savedbitsbytelength = 0;
    if (num_full_block_in_tb > 0) {
        for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
            bit_count = block_pointer[0];
            total_memaccess += sizeof(unsigned char);
            block_pointer++;

            if (bit_count == 0) {
                ori_prior = (float) prior * absErrBound;

                for (j = 0; j < block_size; j++) {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                }
            } else {
                convertByteArray2IntArray_fast_1b_args(
                    block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((block_size - 1) / 8 + 1);
                total_memaccess += (sizeof(unsigned int) * block_size);
                total_memaccess += (sizeof(unsigned char) * ((block_size - 1) / 8 + 1));
                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                    block_pointer, block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                total_memaccess += (sizeof(unsigned int) * block_size);
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
                for (j = 0; j < block_size; j++) {
                    if (temp_sign_arr[j] == 0) {
                        diff = temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    } else {
                        diff = 0 - temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    current = prior + diff;
                    ori_current = (float) current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                    newData_perthread++;
                }
            }
        }
    }

    if (num_remainder_in_tb > 0) {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
            bit_count = block_pointer[0];
            total_memaccess += sizeof(unsigned char);
            block_pointer++;
            if (bit_count == 0) {
                ori_prior = (float) prior * absErrBound;
                for (j = 0; j < num_remainder_in_tb; j++) {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                }
            } else {
                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                    block_pointer,
                    (num_remainder_in_tb - 1) / 8 + 1,
                    temp_sign_arr);
                block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * ((num_remainder_in_tb - 1) / 8 + 1));

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                    block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
                for (j = 0; j < num_remainder_in_tb; j++) {
                    if (temp_sign_arr[j] == 0) {
                        diff = temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    } else {
                        diff = 0 - temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    current = prior + diff;
                    ori_current = (float) current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                }
            }
        }
    }
    free(temp_predict_arr);
    free(temp_sign_arr);

    return total_memaccess;
}

void ZCCL_float_decompress_openmp_threadblock_arg(float *newData,
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

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
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

void ZCCL_float_decompress_openmp_threadblock_randomaccess(float **newData,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes)
{
#ifdef _OPENMP
    *newData = (float *) malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *) cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

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
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            num_full_block_in_tb = (threadblocksize) / block_size;
            num_remainder_in_tb = (threadblocksize) % block_size;
        }
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;
        float *newData_perthread = *newData + lo;
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid];
        unsigned char *block_pointer = outputBytes_perthread;

        float ori_prior = 0.0;
        float ori_current = 0.0;

        unsigned char *temp_sign_arr =
            (unsigned char *) malloc((new_block_size) * sizeof(unsigned char));

        unsigned int *temp_predict_arr =
            (unsigned int *) malloc((new_block_size) * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;
        if (num_full_block_in_tb > 0) {
            for (i = lo; i < hi - num_remainder_in_tb; i = i + block_size) {
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(unsigned int);
                ori_prior = (float) prior * absErrBound;
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                newData_perthread += 1;

                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0) {
                    ori_prior = (float) prior * absErrBound;

                    for (j = 0; j < new_block_size; j++) {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                } else {
                    convertByteArray2IntArray_fast_1b_args(
                        new_block_size, block_pointer, (new_block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((new_block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, new_block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < new_block_size; j++) {
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
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(unsigned int);
                ori_prior = (float) prior * absErrBound;
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                newData_perthread += 1;

                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count == 0) {
                    ori_prior = (float) prior * absErrBound;
                    for (j = 1; j < num_remainder_in_tb; j++) {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                } else {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb - 1,
                        block_pointer,
                        (num_remainder_in_tb - 1 - 1) / 8 + 1,
                        temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1 - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, num_remainder_in_tb - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < num_remainder_in_tb - 1; j++) {
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
            unsigned int num_full_block_in_rm = (remainder) / block_size;
            unsigned int num_remainder_in_rm = (remainder) % block_size;

            if (num_full_block_in_rm > 0) {
                for (i = hi; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    memcpy(&prior, block_pointer, sizeof(int));
                    block_pointer += sizeof(int);
                    ori_prior = (float) prior * absErrBound;
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0) {
                        ori_prior = (float) prior * absErrBound;
                        for (j = 0; j < new_block_size; j++) {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    } else {
                        convertByteArray2IntArray_fast_1b_args(new_block_size,
                            block_pointer,
                            (new_block_size - 1) / 8 + 1,
                            temp_sign_arr);
                        block_pointer += ((new_block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, new_block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < new_block_size; j++) {
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
                    memcpy(&prior, block_pointer, sizeof(int));
                    block_pointer += sizeof(int);
                    ori_prior = (float) prior * absErrBound;
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0) {
                        ori_prior = (float) prior * absErrBound;
                        for (j = 0; j < num_remainder_in_rm - 1; j++) {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    } else {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm - 1,
                            block_pointer,
                            (num_remainder_in_rm - 1 - 1) / 8 + 1,
                            temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1 - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, num_remainder_in_rm - 1, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < num_remainder_in_rm - 1; j++) {
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
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void ZCCL_float_homomophic_add_openmp_threadblock(unsigned char *final_cmpBytes,
    size_t *final_cmpSize,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes,
    unsigned char *cmpBytes2)
{
#ifdef _OPENMP

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *final_real_outputBytes;
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *) cmpBytes;
    size_t *offsets2 = (size_t *) cmpBytes2;
    size_t *offsets_sum = (size_t *) final_cmpBytes;

    unsigned char *rcp;
    unsigned char *rcp2;
    unsigned char *rcp_sum;

    unsigned int nbThreads = 0;

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
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            rcp2 = cmpBytes2 + nbThreads * sizeof(size_t);
            rcp_sum = final_cmpBytes + nbThreads * sizeof(size_t);
            (*final_cmpSize) += nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            num_full_block_in_tb = (threadblocksize - 1) / block_size;
            num_remainder_in_tb = (threadblocksize - 1) % block_size;
            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        }
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;

        size_t i = 0;
        size_t j = 0;
        size_t k = 0;

        unsigned char *final_outputBytes_perthread =
            (unsigned char *) malloc(maxPreservedBufferSize_perthread);
        size_t final_outSize_perthread = 0;

        unsigned char *block_pointer_sum = final_outputBytes_perthread;

        int prior = 0;
        int current = 0;
        int diff = 0;
        int prior2 = 0;
        int current2 = 0;
        int diff2 = 0;

        int prior_sum = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid];
        unsigned char *block_pointer = outputBytes_perthread;
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        unsigned int max2 = 0;
        unsigned int bit_count2 = 0;
        unsigned char *outputBytes_perthread2 = rcp2 + offsets2[tid];
        unsigned char *block_pointer2 = outputBytes_perthread2;
        memcpy(&prior2, block_pointer2, sizeof(int));
        block_pointer2 += sizeof(unsigned int);

        unsigned int bit_count_sum = 0;
        prior_sum = prior + prior2;
        memcpy(block_pointer_sum, &prior_sum, sizeof(int));
        block_pointer_sum += sizeof(unsigned int);
        final_outSize_perthread += sizeof(unsigned int);

        unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;

        unsigned char *temp_sign_arr2 = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr2 = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength2 = 0;
        unsigned int savedbitsbytelength2 = 0;
        int byte_count = 0;
        int remainder_bit = 0;
        size_t byte_offset;

        if (num_full_block_in_tb > 0) {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;

                if (bit_count == 0 && bit_count2 == 0) {
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else if (bit_count == 0 && bit_count2 != 0) {
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength2 = (block_size - 1) / 8 + 1;

                    byte_count = bit_count2 / 8;
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength2 = byte_offset;
                    } else {
                        savedbitsbytelength2 =
                            byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(
                        block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;

                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                } else if (bit_count != 0 && bit_count2 == 0) {
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = (block_size - 1) / 8 + 1;

                    byte_count = bit_count / 8;
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength = byte_offset;
                    } else {
                        savedbitsbytelength =
                            byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;

                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                } else {
                    convertByteArray2IntArray_fast_1b_args(
                        block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(
                        block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                        block_pointer2, block_size, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < block_size; j++) {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                            diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                            diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                        } else {
                            diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        }
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
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else {
                        bit_count_sum = (int) (log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, blockSize, block_pointer_sum);
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        if (num_remainder_in_tb > 0) {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                if (bit_count == 0 && bit_count2 == 0) {
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else if (bit_count == 0 && bit_count2 != 0) {
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;

                    byte_count = bit_count2 / 8;
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength2 = byte_offset;
                    } else {
                        savedbitsbytelength2 = byte_count * num_remainder_in_tb
                            + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(
                        block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;

                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                } else if (bit_count != 0 && bit_count2 == 0) {
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = (num_remainder_in_tb - 1) / 8 + 1;

                    byte_count = bit_count / 8;
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength = byte_offset;
                    } else {
                        savedbitsbytelength = byte_count * num_remainder_in_tb
                            + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;

                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                } else {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                        block_pointer,
                        (num_remainder_in_tb - 1) / 8 + 1,
                        temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                        block_pointer2,
                        (num_remainder_in_tb - 1) / 8 + 1,
                        temp_sign_arr2);
                    block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                        block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < num_remainder_in_tb; j++) {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                            diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                            diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                        } else {
                            diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        }
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
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else {
                        bit_count_sum = (int) (log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, num_remainder_in_tb, block_pointer_sum);
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr,
                            num_remainder_in_tb,
                            block_pointer_sum,
                            bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        if (tid == nbThreads - 1 && remainder != 0) {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size;
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            memcpy(&prior2, block_pointer2, sizeof(int));
            block_pointer2 += sizeof(unsigned int);
            prior_sum = prior + prior2;
            memcpy(block_pointer_sum, &prior_sum, sizeof(int));
            block_pointer_sum += sizeof(unsigned int);
            final_outSize_perthread += sizeof(unsigned int);

            if (num_full_block_in_rm > 0) {
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0) {
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else if (bit_count == 0 && bit_count2 != 0) {
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength2 = (block_size - 1) / 8 + 1;

                        byte_count = bit_count2 / 8;
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength2 = byte_offset;
                        } else {
                            savedbitsbytelength2 =
                                byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum,
                            block_pointer2,
                            signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;

                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    } else if (bit_count != 0 && bit_count2 == 0) {
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = (block_size - 1) / 8 + 1;

                        byte_count = bit_count / 8;
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength = byte_offset;
                        } else {
                            savedbitsbytelength =
                                byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(
                            block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;

                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    } else {
                        convertByteArray2IntArray_fast_1b_args(
                            block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(
                            block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                            block_pointer2, block_size, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < block_size; j++) {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                                diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                                diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                            } else {
                                diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            }
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
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        } else {
                            bit_count_sum = (int) (log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(
                                temp_sign_arr, blockSize, block_pointer_sum);
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(
                                temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0) {
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0) {
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else if (bit_count == 0 && bit_count2 != 0) {
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength2 = (num_remainder_in_rm - 1) / 8 + 1;

                        byte_count = bit_count2 / 8;
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength2 = byte_offset;
                        } else {
                            savedbitsbytelength2 = byte_count * num_remainder_in_rm
                                + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum,
                            block_pointer2,
                            signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;

                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    } else if (bit_count != 0 && bit_count2 == 0) {
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = (num_remainder_in_rm - 1) / 8 + 1;

                        byte_count = bit_count / 8;
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength = byte_offset;
                        } else {
                            savedbitsbytelength = byte_count * num_remainder_in_rm
                                + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(
                            block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;

                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    } else {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm,
                            block_pointer,
                            (num_remainder_in_rm - 1) / 8 + 1,
                            temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm,
                            block_pointer2,
                            (num_remainder_in_rm - 1) / 8 + 1,
                            temp_sign_arr2);
                        block_pointer2 += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                            block_pointer2, num_remainder_in_rm, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < num_remainder_in_rm; j++) {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                                diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                                diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                            } else {
                                diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            }
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
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        } else {
                            bit_count_sum = (int) (log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(
                                temp_sign_arr, num_remainder_in_rm, block_pointer_sum);
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr,
                                num_remainder_in_rm,
                                block_pointer_sum,
                                bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
        }

        offsets_sum[tid] = final_outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + offsets_sum[i - 1];
            }
            (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
            memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);
        }

        memcpy(final_real_outputBytes + offsets_perthread_arr[tid],
            final_outputBytes_perthread,
            final_outSize_perthread);
#pragma omp barrier

        free(final_outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_sign_arr2);
        free(temp_predict_arr2);
#pragma omp single
        {
            free(offsets_perthread_arr);
        }
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void ZCCL_float_homomophic_add_single_thread(unsigned char *final_cmpBytes,
    size_t *final_cmpSize,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes,
    unsigned char *cmpBytes2)
{
    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *final_real_outputBytes;
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *) cmpBytes;
    size_t *offsets2 = (size_t *) cmpBytes2;
    size_t *offsets_sum = (size_t *) final_cmpBytes;

    unsigned char *rcp;
    unsigned char *rcp2;
    unsigned char *rcp_sum;

    unsigned int nbThreads = 0;

    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = blockSize;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

    nbThreads = 1;
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    rcp2 = cmpBytes2 + nbThreads * sizeof(size_t);
    rcp_sum = final_cmpBytes + nbThreads * sizeof(size_t);
    (*final_cmpSize) += nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;
    remainder = nbEle % nbThreads;
    num_full_block_in_tb = (threadblocksize - 1) / block_size;
    num_remainder_in_tb = (threadblocksize - 1) % block_size;
    maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;

    int tid = 0;
    int lo = tid * threadblocksize;
    int hi = (tid + 1) * threadblocksize;

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;

    unsigned char *final_outputBytes_perthread =
        (unsigned char *) malloc(maxPreservedBufferSize_perthread);
    size_t final_outSize_perthread = 0;

    unsigned char *block_pointer_sum = final_outputBytes_perthread;

    int prior = 0;
    int current = 0;
    int diff = 0;
    int prior2 = 0;
    int current2 = 0;
    int diff2 = 0;

    int prior_sum = 0;

    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *outputBytes_perthread = rcp + offsets[tid];
    unsigned char *block_pointer = outputBytes_perthread;
    memcpy(&prior, block_pointer, sizeof(int));
    block_pointer += sizeof(unsigned int);

    unsigned int max2 = 0;
    unsigned int bit_count2 = 0;
    unsigned char *outputBytes_perthread2 = rcp2 + offsets2[tid];
    unsigned char *block_pointer2 = outputBytes_perthread2;
    memcpy(&prior2, block_pointer2, sizeof(int));
    block_pointer2 += sizeof(unsigned int);

    unsigned int bit_count_sum = 0;
    prior_sum = prior + prior2;
    memcpy(block_pointer_sum, &prior_sum, sizeof(int));
    block_pointer_sum += sizeof(unsigned int);
    final_outSize_perthread += sizeof(unsigned int);

    unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

    unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0;
    unsigned int savedbitsbytelength = 0;

    unsigned char *temp_sign_arr2 = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

    unsigned int *temp_predict_arr2 = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength2 = 0;
    unsigned int savedbitsbytelength2 = 0;
    int byte_count = 0;
    int remainder_bit = 0;
    size_t byte_offset;

    if (num_full_block_in_tb > 0) {
        for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
            bit_count = block_pointer[0];
            block_pointer++;
            bit_count2 = block_pointer2[0];
            block_pointer2++;

            if (bit_count == 0 && bit_count2 == 0) {
                block_pointer_sum[0] = 0;
                block_pointer_sum++;
                final_outSize_perthread++;
            } else if (bit_count == 0 && bit_count2 != 0) {
                block_pointer_sum[0] = bit_count2;
                block_pointer_sum++;
                final_outSize_perthread++;

                signbytelength2 = (block_size - 1) / 8 + 1;

                byte_count = bit_count2 / 8;
                remainder_bit = bit_count2 % 8;
                if (remainder_bit == 0) {
                    byte_offset = byte_count * block_size;
                    savedbitsbytelength2 = byte_offset;
                } else {
                    savedbitsbytelength2 =
                        byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                block_pointer_sum += signbytelength2;
                block_pointer_sum += savedbitsbytelength2;
                final_outSize_perthread += signbytelength2;
                final_outSize_perthread += savedbitsbytelength2;

                block_pointer2 += signbytelength2;
                block_pointer2 += savedbitsbytelength2;
            } else if (bit_count != 0 && bit_count2 == 0) {
                block_pointer_sum[0] = bit_count;
                block_pointer_sum++;
                final_outSize_perthread++;

                signbytelength = (block_size - 1) / 8 + 1;

                byte_count = bit_count / 8;
                remainder_bit = bit_count % 8;
                if (remainder_bit == 0) {
                    byte_offset = byte_count * block_size;
                    savedbitsbytelength = byte_offset;
                } else {
                    savedbitsbytelength =
                        byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                block_pointer_sum += signbytelength;
                block_pointer_sum += savedbitsbytelength;
                final_outSize_perthread += signbytelength;
                final_outSize_perthread += savedbitsbytelength;

                block_pointer += signbytelength;
                block_pointer += savedbitsbytelength;
            } else {
                convertByteArray2IntArray_fast_1b_args(
                    block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                    block_pointer, block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                convertByteArray2IntArray_fast_1b_args(
                    block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                block_pointer2 += ((block_size - 1) / 8 + 1);

                savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                    block_pointer2, block_size, temp_predict_arr2, bit_count2);
                block_pointer2 += savedbitsbytelength2;

                max = 0;
                for (j = 0; j < block_size; j++) {
                    if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                        diff = temp_predict_arr[j] + temp_predict_arr2[j];
                    } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                        diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                    } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                        diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                    } else {
                        diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                    }
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
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else {
                    bit_count_sum = (int) (log2f(max)) + 1;
                    block_pointer_sum[0] = bit_count_sum;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = convertIntArray2ByteArray_fast_1b_args(
                        temp_sign_arr, blockSize, block_pointer_sum);
                    block_pointer_sum += signbytelength;
                    final_outSize_perthread += signbytelength;

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
                        temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                }
            }
        }
    }

    if (num_remainder_in_tb > 0) {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
            bit_count = block_pointer[0];
            block_pointer++;
            bit_count2 = block_pointer2[0];
            block_pointer2++;
            if (bit_count == 0 && bit_count2 == 0) {
                block_pointer_sum[0] = 0;
                block_pointer_sum++;
                final_outSize_perthread++;
            } else if (bit_count == 0 && bit_count2 != 0) {
                block_pointer_sum[0] = bit_count2;
                block_pointer_sum++;
                final_outSize_perthread++;

                signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;

                byte_count = bit_count2 / 8;
                remainder_bit = bit_count2 % 8;
                if (remainder_bit == 0) {
                    byte_offset = byte_count * num_remainder_in_tb;
                    savedbitsbytelength2 = byte_offset;
                } else {
                    savedbitsbytelength2 = byte_count * num_remainder_in_tb
                        + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                block_pointer_sum += signbytelength2;
                block_pointer_sum += savedbitsbytelength2;
                final_outSize_perthread += signbytelength2;
                final_outSize_perthread += savedbitsbytelength2;

                block_pointer2 += signbytelength2;
                block_pointer2 += savedbitsbytelength2;
            } else if (bit_count != 0 && bit_count2 == 0) {
                block_pointer_sum[0] = bit_count;
                block_pointer_sum++;
                final_outSize_perthread++;

                signbytelength = (num_remainder_in_tb - 1) / 8 + 1;

                byte_count = bit_count / 8;
                remainder_bit = bit_count % 8;
                if (remainder_bit == 0) {
                    byte_offset = byte_count * num_remainder_in_tb;
                    savedbitsbytelength = byte_offset;
                } else {
                    savedbitsbytelength = byte_count * num_remainder_in_tb
                        + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                block_pointer_sum += signbytelength;
                block_pointer_sum += savedbitsbytelength;
                final_outSize_perthread += signbytelength;
                final_outSize_perthread += savedbitsbytelength;

                block_pointer += signbytelength;
                block_pointer += savedbitsbytelength;
            } else {
                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                    block_pointer,
                    (num_remainder_in_tb - 1) / 8 + 1,
                    temp_sign_arr);
                block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                    block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                    block_pointer2,
                    (num_remainder_in_tb - 1) / 8 + 1,
                    temp_sign_arr2);
                block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                    block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                block_pointer2 += savedbitsbytelength2;

                max = 0;
                for (j = 0; j < num_remainder_in_tb; j++) {
                    if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                        diff = temp_predict_arr[j] + temp_predict_arr2[j];
                    } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                        diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                    } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                        diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                    } else {
                        diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                    }
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
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else {
                    bit_count_sum = (int) (log2f(max)) + 1;
                    block_pointer_sum[0] = bit_count_sum;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = convertIntArray2ByteArray_fast_1b_args(
                        temp_sign_arr, num_remainder_in_tb, block_pointer_sum);
                    block_pointer_sum += signbytelength;
                    final_outSize_perthread += signbytelength;

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
                        temp_predict_arr, num_remainder_in_tb, block_pointer_sum, bit_count_sum);
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                }
            }
        }
    }

    offsets_sum[tid] = final_outSize_perthread;

    offsets_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
    offsets_perthread_arr[0] = 0;

    (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
    memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
    final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);

    memcpy(final_real_outputBytes + offsets_perthread_arr[tid],
        final_outputBytes_perthread,
        final_outSize_perthread);

    free(final_outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(temp_sign_arr2);
    free(temp_predict_arr2);

    free(offsets_perthread_arr);
}

void ZCCL_float_homomophic_add_openmp_threadblock_record(unsigned char *final_cmpBytes,
    size_t *final_cmpSize,
    size_t nbEle,
    float absErrBound,
    int blockSize,
    unsigned char *cmpBytes,
    unsigned char *cmpBytes2,
    int *total_counter_1,
    int *total_counter_2,
    int *total_counter_3,
    int *total_counter_4)
{
#ifdef _OPENMP

    size_t maxPreservedBufferSize = sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *final_real_outputBytes;
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *) cmpBytes;
    size_t *offsets2 = (size_t *) cmpBytes2;
    size_t *offsets_sum = (size_t *) final_cmpBytes;

    unsigned char *rcp;
    unsigned char *rcp2;
    unsigned char *rcp_sum;

    unsigned int nbThreads = 0;

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
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            rcp2 = cmpBytes2 + nbThreads * sizeof(size_t);
            rcp_sum = final_cmpBytes + nbThreads * sizeof(size_t);
            (*final_cmpSize) += nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            num_full_block_in_tb = (threadblocksize - 1) / block_size;
            num_remainder_in_tb = (threadblocksize - 1) % block_size;
            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        }
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;

        size_t i = 0;
        size_t j = 0;
        size_t k = 0;

        unsigned char *final_outputBytes_perthread =
            (unsigned char *) malloc(maxPreservedBufferSize_perthread);
        size_t final_outSize_perthread = 0;

        unsigned char *block_pointer_sum = final_outputBytes_perthread;

        int prior = 0;
        int current = 0;
        int diff = 0;
        int prior2 = 0;
        int current2 = 0;
        int diff2 = 0;

        int prior_sum = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid];
        unsigned char *block_pointer = outputBytes_perthread;
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        unsigned int max2 = 0;
        unsigned int bit_count2 = 0;
        unsigned char *outputBytes_perthread2 = rcp2 + offsets2[tid];
        unsigned char *block_pointer2 = outputBytes_perthread2;
        memcpy(&prior2, block_pointer2, sizeof(int));
        block_pointer2 += sizeof(unsigned int);

        unsigned int bit_count_sum = 0;
        prior_sum = prior + prior2;
        memcpy(block_pointer_sum, &prior_sum, sizeof(int));
        block_pointer_sum += sizeof(unsigned int);
        final_outSize_perthread += sizeof(unsigned int);

        unsigned char *temp_sign_arr = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;

        unsigned char *temp_sign_arr2 = (unsigned char *) malloc(blockSize * sizeof(unsigned char));

        unsigned int *temp_predict_arr2 = (unsigned int *) malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength2 = 0;
        unsigned int savedbitsbytelength2 = 0;
        int byte_count = 0;
        int remainder_bit = 0;
        size_t byte_offset;

        int counter_1 = 0, counter_2 = 0, counter_3 = 0, counter_4 = 0;

        if (num_full_block_in_tb > 0) {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;

                if (bit_count == 0 && bit_count2 == 0) {
                    counter_1++;

                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else if (bit_count == 0 && bit_count2 != 0) {
                    counter_2++;

                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength2 = (block_size - 1) / 8 + 1;

                    byte_count = bit_count2 / 8;
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength2 = byte_offset;
                    } else {
                        savedbitsbytelength2 =
                            byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(
                        block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;

                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                } else if (bit_count != 0 && bit_count2 == 0) {
                    counter_3++;

                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = (block_size - 1) / 8 + 1;

                    byte_count = bit_count / 8;
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength = byte_offset;
                    } else {
                        savedbitsbytelength =
                            byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;

                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                } else {
                    counter_4++;
                    convertByteArray2IntArray_fast_1b_args(
                        block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(
                        block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                        block_pointer2, block_size, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < block_size; j++) {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                            diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                            diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                        } else {
                            diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        }
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
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else {
                        bit_count_sum = (int) (log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, blockSize, block_pointer_sum);
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        if (num_remainder_in_tb > 0) {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size) {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                if (bit_count == 0 && bit_count2 == 0) {
                    counter_1++;

                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                } else if (bit_count == 0 && bit_count2 != 0) {
                    counter_2++;

                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;

                    byte_count = bit_count2 / 8;
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength2 = byte_offset;
                    } else {
                        savedbitsbytelength2 = byte_count * num_remainder_in_tb
                            + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(
                        block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;

                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                } else if (bit_count != 0 && bit_count2 == 0) {
                    counter_3++;
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = (num_remainder_in_tb - 1) / 8 + 1;

                    byte_count = bit_count / 8;
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0) {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength = byte_offset;
                    } else {
                        savedbitsbytelength = byte_count * num_remainder_in_tb
                            + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;

                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                } else {
                    counter_4++;
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                        block_pointer,
                        (num_remainder_in_tb - 1) / 8 + 1,
                        temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                        block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb,
                        block_pointer2,
                        (num_remainder_in_tb - 1) / 8 + 1,
                        temp_sign_arr2);
                    block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                        block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < num_remainder_in_tb; j++) {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                            diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                            diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                        } else {
                            diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                        }
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
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else {
                        bit_count_sum = (int) (log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, num_remainder_in_tb, block_pointer_sum);
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr,
                            num_remainder_in_tb,
                            block_pointer_sum,
                            bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        if (tid == nbThreads - 1 && remainder != 0) {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size;
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            memcpy(&prior2, block_pointer2, sizeof(int));
            block_pointer2 += sizeof(unsigned int);
            prior_sum = prior + prior2;
            memcpy(block_pointer_sum, &prior_sum, sizeof(int));
            block_pointer_sum += sizeof(unsigned int);
            final_outSize_perthread += sizeof(unsigned int);

            if (num_full_block_in_rm > 0) {
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0) {
                        counter_1++;

                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else if (bit_count == 0 && bit_count2 != 0) {
                        counter_2++;

                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength2 = (block_size - 1) / 8 + 1;

                        byte_count = bit_count2 / 8;
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength2 = byte_offset;
                        } else {
                            savedbitsbytelength2 =
                                byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum,
                            block_pointer2,
                            signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;

                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    } else if (bit_count != 0 && bit_count2 == 0) {
                        counter_3++;

                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = (block_size - 1) / 8 + 1;

                        byte_count = bit_count / 8;
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength = byte_offset;
                        } else {
                            savedbitsbytelength =
                                byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(
                            block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;

                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    } else {
                        counter_4++;
                        convertByteArray2IntArray_fast_1b_args(
                            block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(
                            block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                            block_pointer2, block_size, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < block_size; j++) {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                                diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                                diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                            } else {
                                diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            }
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
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        } else {
                            bit_count_sum = (int) (log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(
                                temp_sign_arr, blockSize, block_pointer_sum);
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(
                                temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0) {
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size) {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0) {
                        counter_1++;

                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    } else if (bit_count == 0 && bit_count2 != 0) {
                        counter_2++;

                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength2 = (num_remainder_in_rm - 1) / 8 + 1;

                        byte_count = bit_count2 / 8;
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength2 = byte_offset;
                        } else {
                            savedbitsbytelength2 = byte_count * num_remainder_in_rm
                                + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum,
                            block_pointer2,
                            signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;

                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    } else if (bit_count != 0 && bit_count2 == 0) {
                        counter_3++;
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = (num_remainder_in_rm - 1) / 8 + 1;

                        byte_count = bit_count / 8;
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0) {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength = byte_offset;
                        } else {
                            savedbitsbytelength = byte_count * num_remainder_in_rm
                                + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(
                            block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;

                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    } else {
                        counter_4++;
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm,
                            block_pointer,
                            (num_remainder_in_rm - 1) / 8 + 1,
                            temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                            block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm,
                            block_pointer2,
                            (num_remainder_in_rm - 1) / 8 + 1,
                            temp_sign_arr2);
                        block_pointer2 += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(
                            block_pointer2, num_remainder_in_rm, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < num_remainder_in_rm; j++) {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0) {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0) {
                                diff = (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            } else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0) {
                                diff = 0 - (int) temp_predict_arr[j] + (int) temp_predict_arr2[j];
                            } else {
                                diff = 0 - (int) temp_predict_arr[j] - (int) temp_predict_arr2[j];
                            }
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
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        } else {
                            bit_count_sum = (int) (log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(
                                temp_sign_arr, num_remainder_in_rm, block_pointer_sum);
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr,
                                num_remainder_in_rm,
                                block_pointer_sum,
                                bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
        }

        offsets_sum[tid] = final_outSize_perthread;

#pragma omp barrier
#pragma omp critical
        {
            *total_counter_1 += counter_1;
            *total_counter_2 += counter_2;
            *total_counter_3 += counter_3;
            *total_counter_4 += counter_4;
        }
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr = (size_t *) malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + offsets_sum[i - 1];
            }
            (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
            memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);
        }

        memcpy(final_real_outputBytes + offsets_perthread_arr[tid],
            final_outputBytes_perthread,
            final_outSize_perthread);
#pragma omp barrier

        free(final_outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_sign_arr2);
        free(temp_predict_arr2);
#pragma omp single
        {
            free(offsets_perthread_arr);
        }
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void SZp_decompress3D(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    double twice_eb = errorBound * 2;
    DSize_3d size;
    DSize3D_init(&size, dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        float * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            float * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(!fixed_rate){
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(unsigned int));
                }else{
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }
                float * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            int s = -(int)signFlag[index];
                            curr_buffer_pos[0] = (absPredError[index] ^ s) - s;
                            index++;
                            recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
                            curr_data_pos[0] = curr_buffer_pos[0] * twice_eb;
                            curr_data_pos++;
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_decompress3D_fast_openmp(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
#ifdef _OPENMP
    size_t nbEle = dim1 * dim2 * dim3;
    size_t gloff_0 = dim2 * dim3;
    size_t gloff_1 = dim3;

    size_t *offsets = (size_t *) cmpData;
    unsigned char *rcp;
    int nbThreads = 0;

    double twice_eb = 2 * errorBound;

    size_t base_x, base_y, base_z;
    size_t rem_x, rem_y, rem_z;

    Meta *meta = NULL;
    size_t gloffset;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpData + nbThreads * sizeof(size_t);
        }
#pragma omp barrier
        unsigned char *outputBytes_perthread = rcp + offsets[tid];

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

                    meta[r++] = (Meta){ox, oy, oz, sx, sy, sz};
                }
            }
        }
        int cx = tid / (tdims[1] * tdims[2]);
        int cy = (tid % (tdims[1] * tdims[2])) / tdims[2];
        int cz = tid % tdims[2];

        Meta m = meta[tid];
        gloffset = m.ox * gloff_0 + m.oy * gloff_1 + m.oz;

        DSize_3d size;
        DSize3D_init(&size, m.sx, m.sy, m.sz, blockSideLength);

        size_t lcoff_0 = (size.dim2 + 1) * (size.dim3 + 1);
        size_t lcoff_1 = size.dim3 + 1;
        int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
        memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
        unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
        unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
        unsigned char * thr_pos = outputBytes_perthread + size.num_blocks * FIXED_RATE_PER_BLOCK_BYTES;
        int * prefix = (int *)malloc((size.dim2+1)*(size.dim3+1)*sizeof(int));
        memset(prefix, 0, (size.dim2+1)*(size.dim3+1)*sizeof(int));
        int * colSum = (int *)malloc((size.dim2)*(size.dim3)*sizeof(int));
        memset(colSum, 0, (size.dim2)*(size.dim3)*sizeof(int));
        int block_ind = 0;
        for(size_t x=0; x<size.block_dim1; x++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int * buffer_start_pos = quant_buffer + lcoff_0 + lcoff_1 + 1;
            for(size_t y=0; y<size.block_dim2; y++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                for(size_t z=0; z<size.block_dim3; z++){
                    int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                    int block_size = size_x * size_y * size_z;
                    int fixed_rate = (int)outputBytes_perthread[block_ind++];
                    int * block_buffer_pos = buffer_start_pos;
                    if(fixed_rate){
                        size_t cmp_block_sign_length = (block_size + 7) / 8;
                        convertByteArray2IntArray_fast_1b_args(block_size, thr_pos, cmp_block_sign_length, signFlag);
                        thr_pos += cmp_block_sign_length;
                        unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(thr_pos, block_size, absPredError, fixed_rate);
                        thr_pos += savedbitsbytelength;
                        int index = 0;
                        for(int i=0; i<size_x; i++){
                            for(int j=0; j<size_y; j++){
                                int * curr_buffer_pos = block_buffer_pos;
                                for(int k=0; k<size_z; k++){
                                    int s = -(int)signFlag[index];
                                    curr_buffer_pos[k] = (absPredError[index] ^ s) - s;
                                    index++;
                                }
                                block_buffer_pos += lcoff_1;
                            }
                            block_buffer_pos += lcoff_0 - size_y * lcoff_1;
                        }
                    }else{
                        for(int i=0; i<size_x; i++){
                            for(int j=0; j<size_y; j++){
                                int * curr_buffer_pos = block_buffer_pos;
                                for(int k=0; k<size_z; k++){
                                    curr_buffer_pos[k] = 0;
                                }
                                block_buffer_pos += lcoff_1;
                            }
                            block_buffer_pos += lcoff_0 - size_y * lcoff_1;
                        }
                    }
                    buffer_start_pos += size.Bsize;
                }
                buffer_start_pos += size.Bsize * lcoff_1 - size.Bsize * size.block_dim3;
            }
            size_t x_off = (m.ox + x * size.Bsize) * gloff_0;
            size_t y_off = m.oy * gloff_1;
            size_t t_off = x_off + y_off + m.oz;
            int * curr_pos_0 = quant_buffer + lcoff_0 + lcoff_1 + 1;
            for(int i=0; i<size_x; i++){
                for(size_t j = 0; j < size.dim2; j++){
                    int * prefix_above = prefix + j * lcoff_1;
                    int * prefix_curr  = prefix_above + lcoff_1;
                    int row_acc = 0;
                    for(size_t k = 0; k < size.dim3; k++){
                        colSum[j*size.dim3 + k] += curr_pos_0[k];
                        row_acc += colSum[j*size.dim3 + k];
                        int curr = row_acc + prefix_above[k+1];
                        prefix_curr[k+1] = curr;
                        // decData[(m.ox+x*size.Bsize+i)*gloff_0+(m.oy+j)*gloff_1+m.oz+k] = curr * twice_eb;
                        decData[t_off + i * gloff_0 + j * gloff_1 + k] = curr * twice_eb;
                    }
                    curr_pos_0 += lcoff_1;
                }
                curr_pos_0 += lcoff_0 - size.dim2 * lcoff_1;
            }
            memcpy(quant_buffer, quant_buffer+size.Bsize*lcoff_0, lcoff_0*sizeof(int));
        }
#pragma omp barrier
        free(quant_buffer);
        free(absPredError);
        free(signFlag);
        free(colSum);
        free(prefix);
    }
#else
    printf("Error! OpenMP not supported!\n");
#endif
}