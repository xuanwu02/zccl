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

void ZCCL_float_decompress_openmp_threadblock(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
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

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        newData_perthread += 1;
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        if (num_full_block_in_tb > 0)
        {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count >= 32)
                {
                    printf("In decompression: num_full_block_in_tb i %zu, bit_count %u at thread %d\n", i, bit_count, tid);
                }
                
                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    
                    for (j = 0; j < block_size; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    
                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < block_size; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        
        if (num_remainder_in_tb > 0)
        {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }

        
        if (tid == nbThreads - 1 && remainder != 0)
        {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size; 
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
            if (num_full_block_in_rm > 0)
            {
                
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size)
                {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < block_size; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < block_size; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
                            prior = current;
                            memcpy(newData_perthread, &ori_current, sizeof(float));
                            newData_perthread++;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0)
            {
                
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
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

void ZCCL_float_decompress_single_thread_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{

    
    size_t *offsets = (size_t *)cmpBytes;
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

    ori_prior = (float)prior * absErrBound;
    memcpy(newData_perthread, &ori_prior, sizeof(float)); 
    newData_perthread += 1;
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    if (num_full_block_in_tb > 0)
    {
        for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
        {
            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                
                convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                }
            }
        }
    }
    
    if (num_remainder_in_tb > 0)
    {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
        {
            bit_count = block_pointer[0];
            block_pointer++;
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                for (j = 0; j < num_remainder_in_tb; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < num_remainder_in_tb; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
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

size_t ZCCL_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
    size_t total_memaccess = 0;
    
    size_t *offsets = (size_t *)cmpBytes;
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

    ori_prior = (float)prior * absErrBound;
    memcpy(newData_perthread, &ori_prior, sizeof(float)); 
    total_memaccess += sizeof(float);
    total_memaccess += sizeof(float);
    newData_perthread += 1;
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    if (num_full_block_in_tb > 0)
    {
        for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
        {
            bit_count = block_pointer[0];
            total_memaccess += sizeof(unsigned char);
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                }
            }
            else
            {
                
                convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((block_size - 1) / 8 + 1);
                total_memaccess += (sizeof(unsigned int) * block_size);
                total_memaccess += (sizeof(unsigned char) * ((block_size - 1) / 8 + 1));
                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                total_memaccess += (sizeof(unsigned int) * block_size);
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
                for (j = 0; j < block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                    newData_perthread++;
                }
            }
        }
    }
    
    if (num_remainder_in_tb > 0)
    {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
        {
            bit_count = block_pointer[0];
            total_memaccess += sizeof(unsigned char);
            block_pointer++;
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                for (j = 0; j < num_remainder_in_tb; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    total_memaccess += sizeof(float);
                    total_memaccess += sizeof(float);
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * ((num_remainder_in_tb - 1) / 8 + 1));

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                total_memaccess += (sizeof(unsigned int) * num_remainder_in_tb);
                total_memaccess += (sizeof(unsigned char) * savedbitsbytelength);
                for (j = 0; j < num_remainder_in_tb; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                        total_memaccess += sizeof(unsigned int);
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
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

void ZCCL_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    
    size_t *offsets = (size_t *)cmpBytes;
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

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        newData_perthread += 1;
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        if (num_full_block_in_tb > 0)
        {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                
                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    
                    for (j = 0; j < block_size; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    
                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < block_size; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        
        if (num_remainder_in_tb > 0)
        {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }

        
        if (tid == nbThreads - 1 && remainder != 0)
        {
            unsigned int num_full_block_in_rm = (remainder - 1) / block_size; 
            unsigned int num_remainder_in_rm = (remainder - 1) % block_size;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
            if (num_full_block_in_rm > 0)
            {
                
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size)
                {
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < block_size; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < block_size; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
                            prior = current;
                            memcpy(newData_perthread, &ori_current, sizeof(float));
                            newData_perthread++;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0)
            {
                
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
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

void ZCCL_float_decompress_openmp_threadblock_randomaccess(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
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

        
        unsigned char *temp_sign_arr = (unsigned char *)malloc((new_block_size) * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((new_block_size) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        if (num_full_block_in_tb > 0)
        {
            for (i = lo; i < hi - num_remainder_in_tb; i = i + block_size)
            {
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(unsigned int);
                ori_prior = (float)prior * absErrBound;
                memcpy(newData_perthread, &ori_prior, sizeof(float)); 
                newData_perthread += 1;

                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    
                    for (j = 0; j < new_block_size; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    
                    convertByteArray2IntArray_fast_1b_args(new_block_size, block_pointer, (new_block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((new_block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, new_block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < new_block_size; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        
        if (num_remainder_in_tb > 0)
        {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
            {
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(unsigned int);
                ori_prior = (float)prior * absErrBound;
                memcpy(newData_perthread, &ori_prior, sizeof(float)); 
                newData_perthread += 1;

                bit_count = block_pointer[0];
                block_pointer++;
                if (bit_count == 0)
                {
                    ori_prior = (float)prior * absErrBound;
                    for (j = 1; j < num_remainder_in_tb; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb - 1, block_pointer, (num_remainder_in_tb - 1 - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1 - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < num_remainder_in_tb - 1; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }

        
        if (tid == nbThreads - 1 && remainder != 0)
        {
            unsigned int num_full_block_in_rm = (remainder) / block_size; 
            unsigned int num_remainder_in_rm = (remainder) % block_size;

            if (num_full_block_in_rm > 0)
            {
                
                for (i = hi; i < nbEle - num_remainder_in_rm; i = i + block_size)
                {
                    memcpy(&prior, block_pointer, sizeof(int));
                    block_pointer += sizeof(int);
                    ori_prior = (float)prior * absErrBound;
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < new_block_size; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(new_block_size, block_pointer, (new_block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((new_block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, new_block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < new_block_size; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
                            prior = current;
                            memcpy(newData_perthread, &ori_current, sizeof(float));
                            newData_perthread++;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0)
            {
                
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size)
                {
                    memcpy(&prior, block_pointer, sizeof(int));
                    block_pointer += sizeof(int);
                    ori_prior = (float)prior * absErrBound;
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                    bit_count = block_pointer[0];
                    block_pointer++;
                    if (bit_count == 0)
                    {
                        ori_prior = (float)prior * absErrBound;
                        for (j = 0; j < num_remainder_in_rm - 1; j++)
                        {
                            memcpy(newData_perthread, &ori_prior, sizeof(float));
                            newData_perthread++;
                        }
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm - 1, block_pointer, (num_remainder_in_rm - 1 - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1 - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_rm - 1, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;
                        for (j = 0; j < num_remainder_in_rm - 1; j++)
                        {
                            if (temp_sign_arr[j] == 0)
                            {
                                diff = temp_predict_arr[j];
                            }
                            else
                            {
                                diff = 0 - temp_predict_arr[j];
                            }
                            current = prior + diff;
                            ori_current = (float)current * absErrBound;
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

void ZCCL_float_homomophic_add_openmp_threadblock(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2)
{
#ifdef _OPENMP
    
    size_t maxPreservedBufferSize = sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    
    unsigned char *final_real_outputBytes; 
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *)cmpBytes;
    size_t *offsets2 = (size_t *)cmpBytes2;
    size_t *offsets_sum = (size_t *)final_cmpBytes;

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
        
        unsigned char *final_outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
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

        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        unsigned char *temp_sign_arr2 = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr2 = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength2 = 0; 
        unsigned int savedbitsbytelength2 = 0;
        int byte_count = 0;
        int remainder_bit = 0;
        size_t byte_offset;

        if (num_full_block_in_tb > 0)
        {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                
                if (bit_count == 0 && bit_count2 == 0)
                {
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else if (bit_count == 0 && bit_count2 != 0)
                {
                    
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength2 = (block_size - 1) / 8 + 1;
                    
                    byte_count = bit_count2 / 8; 
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength2 = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength2 = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    
                    memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;
                    
                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                }
                else if (bit_count != 0 && bit_count2 == 0)
                {
                    
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength = (block_size - 1) / 8 + 1;
                    

                    byte_count = bit_count / 8; 
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                    
                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, block_size, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < block_size; j++)
                    {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                        {
                            diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                        }
                        else
                        {
                            diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        if (diff == 0)
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        }
                        else if (diff > 0)
                        {
                            temp_sign_arr[j] = 0;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                        else if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) 
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else
                    {
                        bit_count_sum = (int)(log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer_sum); 
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }

                
                
               
            }
        }
        
        if (num_remainder_in_tb > 0)
        {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                if (bit_count == 0 && bit_count2 == 0)
                {
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else if (bit_count == 0 && bit_count2 != 0)
                {
                    
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;
                    

                    byte_count = bit_count2 / 8; 
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength2 = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength2 = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;
                    
                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                }
                else if (bit_count != 0 && bit_count2 == 0)
                {
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength = (num_remainder_in_tb - 1) / 8 + 1;
                    

                    byte_count = bit_count / 8; 
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                    
                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer2, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                        {
                            diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                        }
                        else
                        {
                            diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        if (diff == 0)
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        }
                        else if (diff > 0)
                        {
                            temp_sign_arr[j] = 0;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                        else if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) 
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else
                    {
                        bit_count_sum = (int)(log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, num_remainder_in_tb, block_pointer_sum); 
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, num_remainder_in_tb, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        
        if (tid == nbThreads - 1 && remainder != 0)
        {
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

            if (num_full_block_in_rm > 0)
            {
                
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0)
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else if (bit_count == 0 && bit_count2 != 0)
                    {
                        
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength2 = (block_size - 1) / 8 + 1;
                        

                        byte_count = bit_count2 / 8; 
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength2 = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength2 = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;
                        
                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    }
                    else if (bit_count != 0 && bit_count2 == 0)
                    {
                        
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength = (block_size - 1) / 8 + 1;
                        

                        byte_count = bit_count / 8; 
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                        
                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, block_size, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < block_size; j++)
                        {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                            {
                                diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                            }
                            else
                            {
                                diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            if (diff == 0)
                            {
                                temp_sign_arr[j] = 0;
                                temp_predict_arr[j] = 0;
                            }
                            else if (diff > 0)
                            {
                                temp_sign_arr[j] = 0;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                            else if (diff < 0)
                            {
                                temp_sign_arr[j] = 1;
                                diff = 0 - diff;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                        }
                        if (max == 0) 
                        {
                            
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        }
                        else
                        {
                            bit_count_sum = (int)(log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer_sum); 
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0)
            {
                
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0)
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else if (bit_count == 0 && bit_count2 != 0)
                    {
                        
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength2 = (num_remainder_in_rm - 1) / 8 + 1;
                        

                        byte_count = bit_count2 / 8; 
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength2 = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength2 = byte_count * num_remainder_in_rm + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;
                        
                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    }
                    else if (bit_count != 0 && bit_count2 == 0)
                    {
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength = (num_remainder_in_rm - 1) / 8 + 1;
                        

                        byte_count = bit_count / 8; 
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength = byte_count * num_remainder_in_rm + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                        
                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    }
                    else
                    {
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer2, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, num_remainder_in_rm, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                            {
                                diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                            }
                            else
                            {
                                diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            if (diff == 0)
                            {
                                temp_sign_arr[j] = 0;
                                temp_predict_arr[j] = 0;
                            }
                            else if (diff > 0)
                            {
                                temp_sign_arr[j] = 0;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                            else if (diff < 0)
                            {
                                temp_sign_arr[j] = 1;
                                diff = 0 - diff;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                        }
                        if (max == 0) 
                        {
                            
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        }
                        else
                        {
                            bit_count_sum = (int)(log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, num_remainder_in_rm, block_pointer_sum); 
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, num_remainder_in_rm, block_pointer_sum, bit_count_sum);
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
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + offsets_sum[i - 1];
                
            }
            (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
            memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);
            
        }
        
        memcpy(final_real_outputBytes + offsets_perthread_arr[tid], final_outputBytes_perthread, final_outSize_perthread);
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

void ZCCL_float_homomophic_add_single_thread(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2)
{

    
    size_t maxPreservedBufferSize = sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    
    unsigned char *final_real_outputBytes; 
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *)cmpBytes;
    size_t *offsets2 = (size_t *)cmpBytes2;
    size_t *offsets_sum = (size_t *)final_cmpBytes;

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
    
    unsigned char *final_outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
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

    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    
    unsigned char *temp_sign_arr2 = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    
    unsigned int *temp_predict_arr2 = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength2 = 0; 
    unsigned int savedbitsbytelength2 = 0;
    int byte_count = 0;
    int remainder_bit = 0;
    size_t byte_offset;

    if (num_full_block_in_tb > 0)
    {
        for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
        {
            bit_count = block_pointer[0];
            block_pointer++;
            bit_count2 = block_pointer2[0];
            block_pointer2++;
            
            if (bit_count == 0 && bit_count2 == 0)
            {
                
                block_pointer_sum[0] = 0;
                block_pointer_sum++;
                final_outSize_perthread++;
            }
            else if (bit_count == 0 && bit_count2 != 0)
            {
                
                block_pointer_sum[0] = bit_count2;
                block_pointer_sum++;
                final_outSize_perthread++;
                
                signbytelength2 = (block_size - 1) / 8 + 1;
                
                byte_count = bit_count2 / 8; 
                remainder_bit = bit_count2 % 8;
                if (remainder_bit == 0)
                {
                    byte_offset = byte_count * block_size;
                    savedbitsbytelength2 = byte_offset;
                }
                else
                {
                    savedbitsbytelength2 = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                }

                
                memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                block_pointer_sum += signbytelength2;
                block_pointer_sum += savedbitsbytelength2;
                final_outSize_perthread += signbytelength2;
                final_outSize_perthread += savedbitsbytelength2;
                
                block_pointer2 += signbytelength2;
                block_pointer2 += savedbitsbytelength2;
            }
            else if (bit_count != 0 && bit_count2 == 0)
            {
                
                block_pointer_sum[0] = bit_count;
                block_pointer_sum++;
                final_outSize_perthread++;
                
                signbytelength = (block_size - 1) / 8 + 1;
                

                byte_count = bit_count / 8; 
                remainder_bit = bit_count % 8;
                if (remainder_bit == 0)
                {
                    byte_offset = byte_count * block_size;
                    savedbitsbytelength = byte_offset;
                }
                else
                {
                    savedbitsbytelength = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                block_pointer_sum += signbytelength;
                block_pointer_sum += savedbitsbytelength;
                final_outSize_perthread += signbytelength;
                final_outSize_perthread += savedbitsbytelength;
                
                block_pointer += signbytelength;
                block_pointer += savedbitsbytelength;
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                convertByteArray2IntArray_fast_1b_args(block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                block_pointer2 += ((block_size - 1) / 8 + 1);

                savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, block_size, temp_predict_arr2, bit_count2);
                block_pointer2 += savedbitsbytelength2;

                max = 0;
                for (j = 0; j < block_size; j++)
                {
                    if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                    {
                        diff = temp_predict_arr[j] + temp_predict_arr2[j];
                    }
                    else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                    {
                        diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                    }
                    else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                    {
                        diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                    }
                    else
                    {
                        diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                    }
                    if (diff == 0)
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    }
                    else if (diff > 0)
                    {
                        temp_sign_arr[j] = 0;
                        if (diff > max)
                        {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                    else if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        diff = 0 - diff;
                        if (diff > max)
                        {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                }
                if (max == 0) 
                {
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else
                {
                    bit_count_sum = (int)(log2f(max)) + 1;
                    block_pointer_sum[0] = bit_count_sum;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer_sum); 
                    block_pointer_sum += signbytelength;
                    final_outSize_perthread += signbytelength;

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                }
            }
        }
    }
    
    if (num_remainder_in_tb > 0)
    {
        for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
        {
            bit_count = block_pointer[0];
            block_pointer++;
            bit_count2 = block_pointer2[0];
            block_pointer2++;
            if (bit_count == 0 && bit_count2 == 0)
            {
                
                block_pointer_sum[0] = 0;
                block_pointer_sum++;
                final_outSize_perthread++;
            }
            else if (bit_count == 0 && bit_count2 != 0)
            {
                
                block_pointer_sum[0] = bit_count2;
                block_pointer_sum++;
                final_outSize_perthread++;
                
                signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;
                

                byte_count = bit_count2 / 8; 
                remainder_bit = bit_count2 % 8;
                if (remainder_bit == 0)
                {
                    byte_offset = byte_count * num_remainder_in_tb;
                    savedbitsbytelength2 = byte_offset;
                }
                else
                {
                    savedbitsbytelength2 = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                block_pointer_sum += signbytelength2;
                block_pointer_sum += savedbitsbytelength2;
                final_outSize_perthread += signbytelength2;
                final_outSize_perthread += savedbitsbytelength2;
                
                block_pointer2 += signbytelength2;
                block_pointer2 += savedbitsbytelength2;
            }
            else if (bit_count != 0 && bit_count2 == 0)
            {
                block_pointer_sum[0] = bit_count;
                block_pointer_sum++;
                final_outSize_perthread++;
                
                signbytelength = (num_remainder_in_tb - 1) / 8 + 1;
                

                byte_count = bit_count / 8; 
                remainder_bit = bit_count % 8;
                if (remainder_bit == 0)
                {
                    byte_offset = byte_count * num_remainder_in_tb;
                    savedbitsbytelength = byte_offset;
                }
                else
                {
                    savedbitsbytelength = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                }

                memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                block_pointer_sum += signbytelength;
                block_pointer_sum += savedbitsbytelength;
                final_outSize_perthread += signbytelength;
                final_outSize_perthread += savedbitsbytelength;
                
                block_pointer += signbytelength;
                block_pointer += savedbitsbytelength;
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer2, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr2);
                block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                block_pointer2 += savedbitsbytelength2;

                max = 0;
                for (j = 0; j < num_remainder_in_tb; j++)
                {
                    if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                    {
                        diff = temp_predict_arr[j] + temp_predict_arr2[j];
                    }
                    else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                    {
                        diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                    }
                    else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                    {
                        diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                    }
                    else
                    {
                        diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                    }
                    if (diff == 0)
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    }
                    else if (diff > 0)
                    {
                        temp_sign_arr[j] = 0;
                        if (diff > max)
                        {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                    else if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        diff = 0 - diff;
                        if (diff > max)
                        {
                            max = diff;
                        }
                        temp_predict_arr[j] = diff;
                    }
                }
                if (max == 0) 
                {
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else
                {
                    bit_count_sum = (int)(log2f(max)) + 1;
                    block_pointer_sum[0] = bit_count_sum;
                    block_pointer_sum++;
                    final_outSize_perthread++;

                    signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, num_remainder_in_tb, block_pointer_sum); 
                    block_pointer_sum += signbytelength;
                    final_outSize_perthread += signbytelength;

                    savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, num_remainder_in_tb, block_pointer_sum, bit_count_sum);
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                }
            }
        }
    }

    offsets_sum[tid] = final_outSize_perthread;

    offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
    offsets_perthread_arr[0] = 0;

    (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
    memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
    final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);
    

    memcpy(final_real_outputBytes + offsets_perthread_arr[tid], final_outputBytes_perthread, final_outSize_perthread);
    
    free(final_outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(temp_sign_arr2);
    free(temp_predict_arr2);

    
    free(offsets_perthread_arr);
}

void ZCCL_float_homomophic_add_openmp_threadblock_record(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2, int *total_counter_1, int *total_counter_2, int *total_counter_3, int *total_counter_4)
{
#ifdef _OPENMP
    
    size_t maxPreservedBufferSize = sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    
    unsigned char *final_real_outputBytes; 
    size_t *offsets_perthread_arr;

    size_t *offsets = (size_t *)cmpBytes;
    size_t *offsets2 = (size_t *)cmpBytes2;
    size_t *offsets_sum = (size_t *)final_cmpBytes;

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
        
        unsigned char *final_outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
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

        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        unsigned char *temp_sign_arr2 = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr2 = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength2 = 0; 
        unsigned int savedbitsbytelength2 = 0;
        int byte_count = 0;
        int remainder_bit = 0;
        size_t byte_offset;

        int counter_1 = 0, counter_2 = 0, counter_3 = 0, counter_4 = 0;

        if (num_full_block_in_tb > 0)
        {
            for (i = lo + 1; i < hi - num_remainder_in_tb; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                
                if (bit_count == 0 && bit_count2 == 0)
                {
                    counter_1++;
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else if (bit_count == 0 && bit_count2 != 0)
                {
                    counter_2++;
                    
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength2 = (block_size - 1) / 8 + 1;
                    
                    byte_count = bit_count2 / 8; 
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength2 = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength2 = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    
                    memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;
                    
                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                }
                else if (bit_count != 0 && bit_count2 == 0)
                {
                    counter_3++;
                    
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength = (block_size - 1) / 8 + 1;
                    

                    byte_count = bit_count / 8; 
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * block_size;
                        savedbitsbytelength = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                    
                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                }
                else
                {
                    counter_4++;
                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((block_size - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, block_size, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < block_size; j++)
                    {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                        {
                            diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                        }
                        else
                        {
                            diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        if (diff == 0)
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        }
                        else if (diff > 0)
                        {
                            temp_sign_arr[j] = 0;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                        else if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) 
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else
                    {
                        bit_count_sum = (int)(log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer_sum); 
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }

                
                
                
            }
        }
        
        if (num_remainder_in_tb > 0)
        {
            for (i = hi - num_remainder_in_tb; i < hi; i = i + block_size)
            {
                bit_count = block_pointer[0];
                block_pointer++;
                bit_count2 = block_pointer2[0];
                block_pointer2++;
                if (bit_count == 0 && bit_count2 == 0)
                {
                    counter_1++;
                    
                    block_pointer_sum[0] = 0;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                }
                else if (bit_count == 0 && bit_count2 != 0)
                {
                    counter_2++;
                    
                    block_pointer_sum[0] = bit_count2;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength2 = (num_remainder_in_tb - 1) / 8 + 1;
                    

                    byte_count = bit_count2 / 8; 
                    remainder_bit = bit_count2 % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength2 = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength2 = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                    block_pointer_sum += signbytelength2;
                    block_pointer_sum += savedbitsbytelength2;
                    final_outSize_perthread += signbytelength2;
                    final_outSize_perthread += savedbitsbytelength2;
                    
                    block_pointer2 += signbytelength2;
                    block_pointer2 += savedbitsbytelength2;
                }
                else if (bit_count != 0 && bit_count2 == 0)
                {
                    counter_3++;
                    block_pointer_sum[0] = bit_count;
                    block_pointer_sum++;
                    final_outSize_perthread++;
                    
                    signbytelength = (num_remainder_in_tb - 1) / 8 + 1;
                    

                    byte_count = bit_count / 8; 
                    remainder_bit = bit_count % 8;
                    if (remainder_bit == 0)
                    {
                        byte_offset = byte_count * num_remainder_in_tb;
                        savedbitsbytelength = byte_offset;
                    }
                    else
                    {
                        savedbitsbytelength = byte_count * num_remainder_in_tb + (remainder_bit * num_remainder_in_tb - 1) / 8 + 1;
                    }

                    memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                    block_pointer_sum += signbytelength;
                    block_pointer_sum += savedbitsbytelength;
                    final_outSize_perthread += signbytelength;
                    final_outSize_perthread += savedbitsbytelength;
                    
                    block_pointer += signbytelength;
                    block_pointer += savedbitsbytelength;
                }
                else
                {
                    counter_4++;
                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr);
                    block_pointer += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_tb, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;

                    convertByteArray2IntArray_fast_1b_args(num_remainder_in_tb, block_pointer2, (num_remainder_in_tb - 1) / 8 + 1, temp_sign_arr2);
                    block_pointer2 += ((num_remainder_in_tb - 1) / 8 + 1);

                    savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, num_remainder_in_tb, temp_predict_arr2, bit_count2);
                    block_pointer2 += savedbitsbytelength2;

                    max = 0;
                    for (j = 0; j < num_remainder_in_tb; j++)
                    {
                        if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = temp_predict_arr[j] + temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                        {
                            diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                        {
                            diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                        }
                        else
                        {
                            diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                        }
                        if (diff == 0)
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        }
                        else if (diff > 0)
                        {
                            temp_sign_arr[j] = 0;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                        else if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            diff = 0 - diff;
                            if (diff > max)
                            {
                                max = diff;
                            }
                            temp_predict_arr[j] = diff;
                        }
                    }
                    if (max == 0) 
                    {
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else
                    {
                        bit_count_sum = (int)(log2f(max)) + 1;
                        block_pointer_sum[0] = bit_count_sum;
                        block_pointer_sum++;
                        final_outSize_perthread++;

                        signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, num_remainder_in_tb, block_pointer_sum); 
                        block_pointer_sum += signbytelength;
                        final_outSize_perthread += signbytelength;

                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, num_remainder_in_tb, block_pointer_sum, bit_count_sum);
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                    }
                }
            }
        }

        
        if (tid == nbThreads - 1 && remainder != 0)
        {
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

            if (num_full_block_in_rm > 0)
            {
                
                for (i = hi + 1; i < nbEle - num_remainder_in_rm; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0)
                    {
                        counter_1++;
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else if (bit_count == 0 && bit_count2 != 0)
                    {
                        counter_2++;
                        
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength2 = (block_size - 1) / 8 + 1;
                        

                        byte_count = bit_count2 / 8; 
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength2 = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength2 = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;
                        
                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    }
                    else if (bit_count != 0 && bit_count2 == 0)
                    {
                        counter_3++;
                        
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength = (block_size - 1) / 8 + 1;
                        

                        byte_count = bit_count / 8; 
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * block_size;
                            savedbitsbytelength = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength = byte_count * block_size + (remainder_bit * block_size - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                        
                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    }
                    else
                    {
                        counter_4++;
                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer, (block_size - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, block_size, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(block_size, block_pointer2, (block_size - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((block_size - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, block_size, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < block_size; j++)
                        {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                            {
                                diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                            }
                            else
                            {
                                diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            if (diff == 0)
                            {
                                temp_sign_arr[j] = 0;
                                temp_predict_arr[j] = 0;
                            }
                            else if (diff > 0)
                            {
                                temp_sign_arr[j] = 0;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                            else if (diff < 0)
                            {
                                temp_sign_arr[j] = 1;
                                diff = 0 - diff;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                        }
                        if (max == 0) 
                        {
                            
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        }
                        else
                        {
                            bit_count_sum = (int)(log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, blockSize, block_pointer_sum); 
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, blockSize, block_pointer_sum, bit_count_sum);
                            block_pointer_sum += savedbitsbytelength;
                            final_outSize_perthread += savedbitsbytelength;
                        }
                    }
                }
            }
            if (num_remainder_in_rm > 0)
            {
                
                for (i = nbEle - num_remainder_in_rm; i < nbEle; i = i + block_size)
                {

                    bit_count = block_pointer[0];
                    block_pointer++;
                    bit_count2 = block_pointer2[0];
                    block_pointer2++;
                    if (bit_count == 0 && bit_count2 == 0)
                    {
                        counter_1++;
                        
                        block_pointer_sum[0] = 0;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                    }
                    else if (bit_count == 0 && bit_count2 != 0)
                    {
                        counter_2++;
                        
                        block_pointer_sum[0] = bit_count2;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength2 = (num_remainder_in_rm - 1) / 8 + 1;
                        

                        byte_count = bit_count2 / 8; 
                        remainder_bit = bit_count2 % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength2 = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength2 = byte_count * num_remainder_in_rm + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer2, signbytelength2 + savedbitsbytelength2);
                        block_pointer_sum += signbytelength2;
                        block_pointer_sum += savedbitsbytelength2;
                        final_outSize_perthread += signbytelength2;
                        final_outSize_perthread += savedbitsbytelength2;
                        
                        block_pointer2 += signbytelength2;
                        block_pointer2 += savedbitsbytelength2;
                    }
                    else if (bit_count != 0 && bit_count2 == 0)
                    {
                        counter_3++;
                        block_pointer_sum[0] = bit_count;
                        block_pointer_sum++;
                        final_outSize_perthread++;
                        
                        signbytelength = (num_remainder_in_rm - 1) / 8 + 1;
                        

                        byte_count = bit_count / 8; 
                        remainder_bit = bit_count % 8;
                        if (remainder_bit == 0)
                        {
                            byte_offset = byte_count * num_remainder_in_rm;
                            savedbitsbytelength = byte_offset;
                        }
                        else
                        {
                            savedbitsbytelength = byte_count * num_remainder_in_rm + (remainder_bit * num_remainder_in_rm - 1) / 8 + 1;
                        }

                        memcpy(block_pointer_sum, block_pointer, signbytelength + savedbitsbytelength);
                        block_pointer_sum += signbytelength;
                        block_pointer_sum += savedbitsbytelength;
                        final_outSize_perthread += signbytelength;
                        final_outSize_perthread += savedbitsbytelength;
                        
                        block_pointer += signbytelength;
                        block_pointer += savedbitsbytelength;
                    }
                    else
                    {
                        counter_4++;
                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr);
                        block_pointer += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, num_remainder_in_rm, temp_predict_arr, bit_count);
                        block_pointer += savedbitsbytelength;

                        convertByteArray2IntArray_fast_1b_args(num_remainder_in_rm, block_pointer2, (num_remainder_in_rm - 1) / 8 + 1, temp_sign_arr2);
                        block_pointer2 += ((num_remainder_in_rm - 1) / 8 + 1);

                        savedbitsbytelength2 = Jiajun_extract_fixed_length_bits(block_pointer2, num_remainder_in_rm, temp_predict_arr2, bit_count2);
                        block_pointer2 += savedbitsbytelength2;

                        max = 0;
                        for (j = 0; j < num_remainder_in_rm; j++)
                        {
                            if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = temp_predict_arr[j] + temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] == 0 && temp_sign_arr2[j] != 0)
                            {
                                diff = (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            else if (temp_sign_arr[j] != 0 && temp_sign_arr2[j] == 0)
                            {
                                diff = 0 - (int)temp_predict_arr[j] + (int)temp_predict_arr2[j];
                            }
                            else
                            {
                                diff = 0 - (int)temp_predict_arr[j] - (int)temp_predict_arr2[j];
                            }
                            if (diff == 0)
                            {
                                temp_sign_arr[j] = 0;
                                temp_predict_arr[j] = 0;
                            }
                            else if (diff > 0)
                            {
                                temp_sign_arr[j] = 0;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                            else if (diff < 0)
                            {
                                temp_sign_arr[j] = 1;
                                diff = 0 - diff;
                                if (diff > max)
                                {
                                    max = diff;
                                }
                                temp_predict_arr[j] = diff;
                            }
                        }
                        if (max == 0) 
                        {
                            
                            block_pointer_sum[0] = 0;
                            block_pointer_sum++;
                            final_outSize_perthread++;
                        }
                        else
                        {
                            bit_count_sum = (int)(log2f(max)) + 1;
                            block_pointer_sum[0] = bit_count_sum;
                            block_pointer_sum++;
                            final_outSize_perthread++;

                            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, num_remainder_in_rm, block_pointer_sum); 
                            block_pointer_sum += signbytelength;
                            final_outSize_perthread += signbytelength;

                            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, num_remainder_in_rm, block_pointer_sum, bit_count_sum);
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
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + offsets_sum[i - 1];
                
            }
            (*final_cmpSize) += offsets_perthread_arr[nbThreads - 1] + offsets_sum[nbThreads - 1];
            memcpy(final_cmpBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            final_real_outputBytes = final_cmpBytes + nbThreads * sizeof(size_t);
            
        }
        
        memcpy(final_real_outputBytes + offsets_perthread_arr[tid], final_outputBytes_perthread, final_outSize_perthread);
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
