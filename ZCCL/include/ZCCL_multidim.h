#ifndef ZCCL_multidim_H
#define ZCCL_multidim_H

#include <stdio.h>
#include "mpi.h"
#include "ZCCL.h"

void pack_block(const float *M,
                       size_t off_0, size_t off_1,
                       int ox, int oy, int oz,
                       int bx, int by, int bz,
                       float *buf);

void distribute3d(const float *M,
                  float *local,
                  int N1, int N2, int N3,
                  MPI_Comm grid,
                  const int dims[3]);                       

typedef struct {
    size_t dim1, dim2, dim3;
    size_t nbEle;

    int    Bsize;
    int    max_num_block_elements;

    size_t block_dim1, block_dim2, block_dim3;
    size_t num_blocks;

    size_t offset_0, offset_1;
} DSize_3d;

static inline void DSize3D_init(DSize_3d *s,
                                size_t r1, size_t r2, size_t r3,
                                int    bs)
{
    s->dim1 = r1;  s->dim2 = r2;  s->dim3 = r3;
    s->nbEle = r1 * r2 * r3;

    s->Bsize  = bs;
    s->max_num_block_elements = bs * bs * bs;

    s->block_dim1 = (r1 - 1) / bs + 1;
    s->block_dim2 = (r2 - 1) / bs + 1;
    s->block_dim3 = (r3 - 1) / bs + 1;
    s->num_blocks = s->block_dim1 * s->block_dim2 * s->block_dim3;

    s->offset_0 = r2 * r3;
    s->offset_1 = r3;
}

int compare_desc(const void *a, const void *b);

void My_Dims_create(int nnodes, int ndims, int dims[]);

#endif