/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Feb, 2024
 */

void get_4_digits(int input, int *output)
{
    if (input < 10)
    {
        output[0] = 0;
        output[1] = 0;
        output[2] = 0;
        output[3] = input;
    }
    else if (input < 100)
    {
        output[0] = 0;
        output[1] = 0;
        output[2] = input / 10;
        output[3] = input % 10;
    }
    else if (input < 1000)
    {
        output[0] = 0;
        output[1] = input / 100;
        output[2] = input / 10 - input / 100 * 10;
        output[3] = input % 10;
    }
    else if (input < 10000)
    {
        output[0] = input / 1000;
        output[1] = input / 100 - input / 1000 * 10;
        output[2] = input / 10 - input / 100 * 10;
        output[3] = input % 10;
    }
}

typedef float data_type;


bool verify_arrays(data_type *array1, data_type *array2, int n)
{
  data_type diff = 0.f;
  int i;

  for (i = 0; i < n; i++)
  {
    diff = fabs(array1[i] - array2[i]);
    if (diff > 1e-4)
    {
      printf("error. %f,%f,%d\n", array1[i], array2[i], i);
      return false;
    }
  }

  return true;
}

data_type *create_rand_nums(int num_elements)
{
  data_type *rand_nums = (data_type *)malloc(sizeof(data_type) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++)
  {
    rand_nums[i] = (rand() / (data_type)RAND_MAX);
  }
  return rand_nums;
}

data_type *create_fixed_nums(int num_elements, int world_rank)
{
  data_type *rand_nums = (data_type *)malloc(sizeof(data_type) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++)
  {
    rand_nums[i] = world_rank + i * 0.1;
  }
  return rand_nums;
}

data_type *inilize_arr(int num_elements)
{
  data_type *rand_nums = (data_type *)malloc(sizeof(data_type) * num_elements);
  assert(rand_nums != NULL);
  memset(rand_nums, 0, sizeof(data_type) * num_elements);
  return rand_nums;
}

void *inilize_arr_withoutset(int num_elements)
{
  void *rand_nums = (void *)malloc(sizeof(data_type) * num_elements);
  assert(rand_nums != NULL);
  return rand_nums;
}

int get_pof2(int num)
{   
    int pof2 = 1;
    while(num > 1)
    {
        num = num/2;
        pof2 *= 2;
    }
    return pof2;
}
