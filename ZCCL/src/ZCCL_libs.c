/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Feb, 2024
 */

#include "ZCCL_libs.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
typedef float data_type;
#define ALL_REDUCE_TAG 14
#define MPIR_ALLGATHERV_TAG 8
#define MPIR_BCAST_TAG 2
#define MPIR_SCATTER_TAG 5
#define MPI_IN_PLACE (void *) -1
#define MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE 42347060
#define MPIR_CVAR_ORI_ALLGATHERV_PIPELINE_MSG_SIZE 32768
#define MPIR_CVAR_ALLGATHERV_CPR_PIPELINE_MSG_SIZE 983040
#define SINGLETHREAD_CHUNK_SIZE 5120
#define MULTITHREAD_CHUNK_SIZE 256000
#define SAVE_CPR_RESULT 1
#define PRINT_DETAILS 0
#define PRINT_EXPLANATION 0
#define PRINT_AVGRESULTS 1
#define PRINT_EXPERIMENTS 1
#define tag_base 100

void add_vectors(float *result, const float *a, const float *b, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
