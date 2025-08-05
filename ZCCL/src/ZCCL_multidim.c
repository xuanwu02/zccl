#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "mpi.h"
#include "ZCCL.h"
#include "ZCCL_TypeManager.h"
#include "ZCCL_BytesToolkit.h"
#include "ZCCL_multidim.h"

void pack_block(const float *M,
                       size_t off_0, size_t off_1,
                       int ox, int oy, int oz,
                       int bx, int by, int bz,
                       float *buf)
{
    int idx = 0;
    for (int x = 0; x < bx; x++) {
        for (int y = 0; y < by; y++) {
            const float *src = M + (size_t)(ox + x) * off_0 + (size_t)(oy + y) * off_1 + oz;
            memcpy(buf + idx, src, bz * sizeof(float));
            idx += bz;
        }
    }
}

void distribute3d(const float *M,
                  float *local,
                  int N1, int N2, int N3,
                  MPI_Comm grid,
                  const int dims[3])
{
    int rank, coords[3];
    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 3, coords); 

    const int base_x = N1 / dims[0], rem_x = N1 % dims[0];
    const int base_y = N2 / dims[1], rem_y = N2 % dims[1];
    const int base_z = N3 / dims[2], rem_z = N3 % dims[2];

    const size_t off_0 = N2 * N3;
    const size_t off_1 = N3;

    if (rank == 0) {
        int ox_acc = 0;
        for (int px = 0; px < dims[0]; ++px) {
            int sx = base_x + (px < rem_x ? 1 : 0);
            int ox = ox_acc;
            ox_acc += sx;

            int oy_acc = 0;
            for (int py = 0; py < dims[1]; ++py) {
                int sy = base_y + (py < rem_y ? 1 : 0);
                int oy = oy_acc;
                oy_acc += sy;

                int oz_acc = 0;
                for (int pz = 0; pz < dims[2]; ++pz) {
                    int sz = base_z + (pz < rem_z ? 1 : 0);
                    int oz = oz_acc;
                    oz_acc += sz;

                    int dest = ((px * dims[1]) + py) * dims[2] + pz;

                    if (dest == 0) {
                        pack_block(M, off_0, off_1,
                                   ox, oy, oz,
                                   sx, sy, sz, local);
                    }
                    else {
                        int sizes[3]   = { N1, N2, N3 };
                        int subs [3]   = { sx , sy , sz };
                        int starts[3]  = { ox , oy , oz };

                        MPI_Datatype subarray;
                        MPI_Type_create_subarray(3, sizes, subs, starts,
                                                 MPI_ORDER_C, MPI_FLOAT, &subarray);
                        MPI_Type_commit(&subarray);
                        MPI_Send(M, 1, subarray, dest, 0, grid);
                        MPI_Type_free(&subarray);
                    }
                }
            }
        }
    }
    else {
        int sx = base_x + (coords[0] < rem_x ? 1 : 0);
        int sy = base_y + (coords[1] < rem_y ? 1 : 0);
        int sz = base_z + (coords[2] < rem_z ? 1 : 0);
        MPI_Recv(local, sx * sy * sz, MPI_FLOAT, 0, 0, grid, MPI_STATUS_IGNORE);
    }
}

int compare_desc(const void *a, const void *b) {
    return *(int *)b - *(int *)a;
}

void My_Dims_create(int nnodes, int ndims, int dims[]) {
    for (int i = 0; i < ndims; i++) {
        if (dims[i] == 0)
            dims[i] = 1;
    }

    int product = 1;
    for (int i = 0; i < ndims; i++) {
        product *= dims[i];
    }

    if (nnodes % product != 0) {
        fprintf(stderr, "Error: current dims product does not divide nnodes\n");
        return;
    }

    int remaining = nnodes / product;

    int factors[64];
    int nfactors = 0;
    for (int i = 2; i <= remaining; i++) {
        while (remaining % i == 0) {
            factors[nfactors++] = i;
            remaining /= i;
        }
    }

    qsort(factors, nfactors, sizeof(int), compare_desc);

    for (int i = 0; i < nfactors; i++) {
        int min_idx = 0;
        for (int j = 1; j < ndims; j++) {
            if (dims[j] < dims[min_idx]) {
                min_idx = j;
            }
        }
        dims[min_idx] *= factors[i];
    }

    qsort(dims, ndims, sizeof(int), compare_desc);
}
