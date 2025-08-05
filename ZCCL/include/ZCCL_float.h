/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef _ZCCL_Float_H
#define _ZCCL_Float_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "ZCCL_defines.h"
#include "ZCCL_multidim.h"

unsigned char *
ZCCL_float_openmp_direct_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                             size_t nbEle, int blockSize);

unsigned char *
ZCCL_float_openmp_threadblock_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                                  size_t nbEle, int blockSize);

unsigned char *
ZCCL_float_openmp_threadblock(float *oriData, size_t *outSize, float absErrBound,
                             size_t nbEle, int blockSize);

void ZCCL_float_openmp_threadblock_arg(unsigned char *outputBytes, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize);

void ZCCL_float_single_thread_arg(unsigned char *outputBytes, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize);

size_t ZCCL_float_single_thread_arg_record(unsigned char *outputBytes, float *oriData, size_t *outSize, float absErrBound,
                                       size_t nbEle, int blockSize);

unsigned char *
ZCCL_float_openmp_threadblock_randomaccess(float *oriData, size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize);


void ZCCL_float_single_thread_arg_split_record(unsigned char *outputBytes,
    float *oriData,
    size_t *outSize,
    float absErrBound,
    size_t nbEle,
    int blockSize,
    unsigned char *chunk_arr,
    size_t chunk_iter);

unsigned char *ZCCL_fast_compress_args(int fastMode, int dataType, void *data, size_t *outSize, int errBoundMode, float absErrBound,
                                      float relBoundRatio, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

inline int SZ_quantize(const float data, const double inver_eb)
{
    int q = floor(data * inver_eb + 0.5);
    return q;
}

inline int predict_lorenzo_3d(
    const float *data_pos, int *buffer_pos, double inver_eb,
    size_t offset_0, size_t offset_1
){
    int curr_quant = SZ_quantize(data_pos[0], inver_eb);
    buffer_pos[0] = curr_quant;
    int pred = buffer_pos[-1] + buffer_pos[-offset_1] + buffer_pos[-offset_0] 
            - buffer_pos[-offset_1 - 1] - buffer_pos[-offset_0 - 1] 
            - buffer_pos[-offset_0 - offset_1] + buffer_pos[-offset_0 - offset_1 - 1];
    return curr_quant - pred;
}

inline void recover_lorenzo_3d(
    int *buffer_pos, size_t offset_0, size_t offset_1
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-offset_1] + buffer_pos[-offset_0] 
                    - buffer_pos[-offset_1 - 1] - buffer_pos[-offset_0 - 1] 
                    - buffer_pos[-offset_0 - offset_1] + buffer_pos[-offset_0 - offset_1 - 1];
}

void SZp_compress3D_fast(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize);

void SZp_compress3D(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize);

typedef struct {
    int ox, oy, oz;
    int sx, sy, sz;
} Meta;

void SZp_compress3D_fast_openmp(
    const float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t *cmpSize);

#endif /* ----- #ifndef _ZCCL_Float_H  ----- */
