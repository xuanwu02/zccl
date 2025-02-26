/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Feb, 2024
 */

#ifndef _ZCCL_UTILS_H
#define _ZCCL_UTILS_H

void get_4_digits(int input, int *output);

typedef float data_type;

bool verify_arrays(data_type *array1, data_type *array2, int n);

data_type *create_rand_nums(int num_elements);

data_type *create_fixed_nums(int num_elements, int world_rank);

data_type *inilize_arr(int num_elements);

void *inilize_arr_withoutset(int num_elements);

int get_pof2(int num);

#endif /* ----- #ifndef _ZCCL_UTILS_H  ----- */