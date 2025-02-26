/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdint.h>
#include "mpi.h"
#include "ring2_multithreads.h"
#include "ring2_ho_multithreads.h"
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdbool.h>
#include "./include/utils.h"

#define ITERATIONS_LARGE 100
#define LARGE_MESSAGE_SIZE 1024 * 1024
#define MIN_MESSAGE_LENGTH 1
#define tolerance 0.08
#define MPI_THREAD_MODE MPI_THREAD_FUNNELED
typedef float data_type;

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MODE, &provided);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    if (world_rank == 0 && PRINT_EXPLANATION) {
        printf("Welcome to our ANL_allreduce_benchmark\n");
    }
    int opt;
    int warm_up = 0;
    int num_trials = 1000;
    int validation = 0;
    int select = 1;
    int minimal_size = 4 / sizeof(data_type);
    int maximal_size = 4 * 1024 * 1024 / sizeof(data_type);
    int large_size = LARGE_MESSAGE_SIZE / sizeof(data_type);
    char *inputDire = NULL;
    double compressionRatio = 1E-3;
    int blockSize = 36;
    while ((opt = getopt(argc, argv, "i:w:v:s:l:k:f:b:r:a:")) != EOF) {
        switch (opt) {
            case 'i':
                num_trials = atoi(optarg);
                break;
            case 'w':
                warm_up = atoi(optarg);
                break;
            case 'v':
                validation = atoi(optarg);
                break;
            case 's':
                minimal_size = atoi(optarg) / sizeof(data_type);
                break;
            case 'l':
                maximal_size = atoi(optarg) / sizeof(data_type);
                break;
            case 'k':
                select = atoi(optarg);
                break;
            case 'f':
                inputDire = optarg;
                break;
            case 'b':
                blockSize = atof(optarg);
                break;
            case 'r':
                compressionRatio = atof(optarg);
                break;
            case '?':
                if (world_rank == 0) {
                    printf(
                        "usage is: ./ANL_benchmarks\n"
                        "-i <number of iterations> the default value is 1000\n"
                        "-w <number of warmups> the default value is 0\n"
                        "-v <enable validation> the default value is 0\n"
                        "-s <minimal data size in bytes> the default value is 4 bytes\n"
                        "-l <maximal data size in bytes> the default value is 4 MB\n"
                        "-k <select the kernel> 0 for original allreduce, 1 for our allreduce, 2 for SZx allreduce default is 1\n"
                        "-f <set input file path> \n"
                        "-? printf this message\n");
                    break;
                }

            default:
                exit(1);
        }
    }
    double absErrBound = compressionRatio;

    int status = 0;
    size_t nbEle;
    int *index_array = (int *) malloc(sizeof(int) * 4);

    int start = 0;
    int step = 0;
    int actual_snap_count = 0;

    char oriFilePath[645];
    char oriFileDire[640];

    if (inputDire == NULL) {
        printf("Please input the inputDire!\n");
    } else {
        sprintf(oriFileDire, "%s", inputDire);

        start = 0;
        step = 0;
        get_4_digits(start + step * world_rank, index_array);
        sprintf(oriFilePath, "%s%d%d.dat", oriFileDire, index_array[2], index_array[3]);
    }

    float *numbers = readFloatData(oriFilePath, &nbEle, &status);
    if (status != SZ_SCES) {
        printf("Error: data file %s cannot be read!\n", oriFilePath);
        exit(0);
    }
    if (!world_rank && PRINT_EXPLANATION)
        printf("Original file size: %d\n", nbEle);
    if (num_trials <= 0) {
        printf("Please select a valid number of iterations.\n");
        exit(1);
    }
    if (warm_up < 0) {
        printf("Please select a valid number of warm_up.\n");
        exit(1);
    }
    if (validation != 0 && validation != 1) {
        printf("Please select a valid status of validation.\n");
        exit(1);
    }
    if (minimal_size < MIN_MESSAGE_LENGTH)
        minimal_size = MIN_MESSAGE_LENGTH;
    if (maximal_size < MIN_MESSAGE_LENGTH)
        maximal_size = MIN_MESSAGE_LENGTH;
    if (maximal_size > nbEle)
        maximal_size = nbEle;

    if (world_rank == 0 && PRINT_EXPLANATION) {
        printf("The settings are: %d iterations, %d warmups, validation: %d, "
               "minimal data size: %ld bytes, maximal data size: %ld bytes, kernel: %d\n",
            num_trials,
            warm_up,
            validation,
            minimal_size * sizeof(data_type),
            maximal_size * sizeof(data_type),
            select);
    }

    int size, iterations, i;

    for (size = minimal_size; size <= maximal_size; size *= 2) {
        iterations = num_trials;

        data_type *invec = NULL;

        invec = inilize_arr_withoutset(size);
        memcpy(invec, numbers, sizeof(data_type) * size);

        data_type *inoutvec = NULL;
        srand(time(NULL));
        inoutvec = inilize_arr(size);

        MPI_Barrier(MPI_COMM_WORLD);

        for (i = 0; i < warm_up; i++) {
            if (select == 0) {
                MPI_Allreduce(invec, inoutvec, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            } else if (select == 1) {
                MPI_Allreduce_ZCCL_RI2_mt_oa_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
            } else if (select == 2) {
                MPI_Allreduce_ZCCL_RI2_mt_oa_ho_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
            } else if (select == 3) {
                MPI_Allreduce_ZCCL_RI2_st_oa_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
            } else if (select == 4) {
                MPI_Allreduce_ZCCL_RI2_st_oa_ho_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double MPI_timer = 0.0;
        for (i = 0; i < iterations; i++) {
            if (select == 0) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPI_Allreduce(invec, inoutvec, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            } else if (select == 1) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPI_Allreduce_ZCCL_RI2_mt_oa_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            } else if (select == 2) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPI_Allreduce_ZCCL_RI2_mt_oa_ho_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            } else if (select == 3) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPI_Allreduce_ZCCL_RI2_st_oa_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            } else if (select == 4) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPI_Allreduce_ZCCL_RI2_st_oa_ho_record(invec,
                    inoutvec,
                    compressionRatio,
                    tolerance,
                    blockSize,
                    size,
                    MPI_FLOAT,
                    MPI_SUM,
                    MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double latency = (double) (MPI_timer * 1e6) / iterations;
        double min_time = 0.0;
        double max_time = 0.0;
        double avg_time = 0.0;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        avg_time = avg_time / world_size;
        if (world_rank == 0) {
            if (PRINT_AVGRESULTS) {
                printf(
                    "Compression-accelerated Kernel %d For datasize: %ld bytes, the avg_time is %f us, the max_time is %f us, the min_time is %f us\n",
                    select,
                    size * sizeof(data_type),
                    avg_time,
                    max_time,
                    min_time);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        free(invec);
        free(inoutvec);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(numbers);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
