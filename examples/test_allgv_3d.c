#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "ZCCL.h"
#include "mpi.h"

int main(int argc, char **argv)
{
    // MPI_Init(&argc, &argv);
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    MPI_Barrier(MPI_COMM_WORLD);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    MPI_Comm grid_comm;
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(nranks, 3, dims);
    int periodic[3] = {1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, 0, &grid_comm);
    int coords[3];
    MPI_Cart_coords(grid_comm, rank, 3, coords); 

    MPI_Datatype MPI_SIZE_T;
    MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(size_t), &MPI_SIZE_T);
    MPI_Type_commit(&MPI_SIZE_T);

    int argv_id = 1;
    char fname[256];
    sprintf(fname, "%s", argv[argv_id++]);
    int n1 = atoi(argv[argv_id++]);
    int n2 = atoi(argv[argv_id++]);
    int n3 = atoi(argv[argv_id++]);
    int blockSize = atoi(argv[argv_id++]);
    int Bsize = atoi(argv[argv_id++]);
    int kernel = atoi(argv[argv_id++]);
    int iters = atoi(argv[argv_id++]);
    double errBound = atof(argv[argv_id++]);
    size_t nbEle = n1 * n2 * n3;

    int data_dims[3] = {n1, n2, n3};
    const int base_x = n1 / dims[0], rem_x = n1 % dims[0];
    const int base_y = n2 / dims[1], rem_y = n2 % dims[1];
    const int base_z = n3 / dims[2], rem_z = n3 % dims[2];
    const int bx = base_x + (coords[0] < rem_x ? 1 : 0);
    const int by = base_y + (coords[1] < rem_y ? 1 : 0);
    const int bz = base_z + (coords[2] < rem_z ? 1 : 0);

    float * buffer = (float *)malloc(nbEle * sizeof(float));

    int status;
    float * A = NULL;
    if (rank == 0){
        size_t num;
        A = readFloatData(fname, &num, &status);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int counts[nranks];
    int displs[nranks];

    Meta *meta = (Meta*)malloc(nranks * sizeof(Meta));

    int pref = 0, r = 0;
    int ox_acc = 0;
    for (int px = 0; px < dims[0]; px++) {
        int sx = base_x + (px < rem_x ? 1 : 0);
        int ox = ox_acc;
        ox_acc += sx;

        int oy_acc = 0;
        for (int py = 0; py < dims[1]; py++) {
            int sy = base_y + (py < rem_y ? 1 : 0);
            int oy = oy_acc;
            oy_acc += sy;

            int oz_acc = 0;
            for (int pz = 0; pz < dims[2]; pz++) {
                int sz = base_z + (pz < rem_z ? 1 : 0);
                int oz = oz_acc;
                oz_acc += sz;

                counts[r]  = sx * sy * sz;
                displs[r]  = pref;
                pref      += counts[r];
                meta[r++] = (Meta){ox, oy, oz, sx, sy, sz};
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    distribute3d(A, buffer + displs[rank], n1, n2, n3, grid_comm, dims);
    MPI_Barrier(MPI_COMM_WORLD);

    // warm up
    int warmup = 10;
    switch(kernel){
        case 0:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
            }
            break;
        }
        case 1:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring_RI2_st_oa_3d(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, data_dims, errBound, 1, Bsize);
            }
            break;
        }
        case 2:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_3d(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, data_dims, errBound, 1, Bsize);
            }
            break;
        }
        case 3:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring_RI2_st_oa_record(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
            }
            break;
        }
        case 4:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
            }
            break;
        }
        case 5:{
            for(int i=0; i<warmup; i++){
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_record_seg(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
            }
            break;
        }
        default:
            break;
    }

    // test
    double MPI_timer = 0.0;
    double avg = 0.0;

    switch(kernel){
        case 0:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("MPI time = %.6f\n", avg/nranks);
            break;
        }
        case 1:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring_RI2_st_oa_3d(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, data_dims, errBound, 1, Bsize);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("zccl_st_3d time = %.6f\n", avg/nranks);
            break;
        }
        case 2:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_3d(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, data_dims, errBound, 1, Bsize);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("zccl_mt_3d time = %.6f\n", avg/nranks);
            break;
        }
        case 3:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring_RI2_st_oa_record(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("zccl_st time = %.6f\n", avg/nranks);
            break;
        }
        case 4:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("zccl_mt time = %.6f\n", avg/nranks);
            break;
        }
        case 5:{
            for(int i=0; i<iters; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_timer -= MPI_Wtime();
                MPIR_Allgatherv_intra_ring_RI2_mt_oa_record_seg(MPI_IN_PLACE, counts[rank], MPI_FLOAT, buffer, counts, displs, MPI_FLOAT, MPI_COMM_WORLD, NULL, errBound, 1, blockSize);
                MPI_timer += MPI_Wtime();
            }
            MPI_timer /= iters;
            MPI_Reduce(&MPI_timer, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(!rank) printf("zccl_mt_seg time = %.6f\n", avg/nranks);
            break;
        }
        default:
            break;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==nranks-1 && iters==1){
        A = (float *)malloc(nbEle * sizeof(float));
        memset(A, 0, nbEle * sizeof(float));
        for (int pid = 0; pid < nranks; pid++) {
            Meta m = meta[pid];
            const float *src = buffer + displs[pid];
            for (int dx = 0; dx < m.sx; dx++) {
                int gx = m.ox + dx;
                for (int dy = 0; dy < m.sy; dy++) {
                    int gy = m.oy + dy;
                    float *dst = A + ((gx * n2 + gy) * n3 + m.oz);
                    memcpy(dst, src, m.sz * sizeof(float));
                    src += m.sz;
                }
            }
        }
        char outputDire[256];
        sprintf(outputDire, "allgv.%d.dat", kernel);
        writeByteData((unsigned char *)A, sizeof(float) * nbEle, outputDire, &status);
        if (status != SZ_SCES) {
            printf("Error: data file %s cannot be written!\n", outputDire);
            exit(0);
        }
        free(A);
    }

    if(!rank) free(A);
    free(buffer);
    MPI_Finalize();

    return 0;
}