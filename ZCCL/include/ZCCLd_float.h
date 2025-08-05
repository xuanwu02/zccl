/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef _ZCCLd_Float_H
#define _ZCCLd_Float_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "ZCCL_defines.h"
#include "ZCCL_multidim.h"

void ZCCL_float_decompress_openmp_threadblock(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void ZCCL_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void ZCCL_float_decompress_single_thread_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

size_t ZCCL_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void ZCCL_float_decompress_openmp_threadblock_randomaccess(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void ZCCL_float_homomophic_add_openmp_threadblock(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2);

void ZCCL_float_homomophic_add_openmp_threadblock_record(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2, int *total_counter_1, int *total_counter_2, int *total_counter_3, int *total_counter_4);

void ZCCL_float_homomophic_add_single_thread(unsigned char *final_cmpBytes, size_t *final_cmpSize, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, unsigned char *cmpBytes2);

void SZp_decompress3D_fast(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound);

void SZp_decompress3D(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound);

void SZp_decompress3D_fast_openmp(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound);

#endif /* ----- #ifndef _ZCCLd_Float_H  ----- */
