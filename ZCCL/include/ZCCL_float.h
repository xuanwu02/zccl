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

unsigned char *ZCCL_fast_compress_args(int fastMode, int dataType, void *data, size_t *outSize, int errBoundMode, float absErrBound,
                                      float relBoundRatio, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

#endif /* ----- #ifndef _ZCCL_Float_H  ----- */
