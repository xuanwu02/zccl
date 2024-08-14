/**
 *  @file ring2_multithreads.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include "mpi.h"
#include "hZCCL.h"

#include "./include/libs.h"


int MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(const void *sendbuf,
                                                MPI_Aint sendcount,
                                                MPI_Datatype sendtype,
                                                void *recvbuf,
                                                const MPI_Aint *recvcounts,
                                                const MPI_Aint *displs,
                                                MPI_Datatype recvtype,
                                                MPI_Comm comm,
                                                unsigned char *outputBytes,
                                                float compressionRatio,
                                                float tolerance,
                                                int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE)
    {
        
        memcpy((char *)recvbuf + displs[rank] * sizeof(recvtype),
               sendbuf, sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;
    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *)malloc(comm_size * sizeof(int));
    
    void *temp_recvbuf = (void *)malloc(total_count * recvtype_extent);

    
    osbuf = (char *)recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *)temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    hZCCL_float_openmp_threadblock_arg(sbuf, osbuf, &outSize, absErrBound,
                                     (recvcounts[sidx]), blockSize);

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);


    
    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0 &&
        max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE)
    {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;
        
        if (!chunk_count)
            chunk_count = 1;
    }
    
    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv)
    {
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) >
                   chunk_count)
                      ? chunk_count
                      : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) >
                   chunk_count)
                      ? chunk_count
                      : (compressed_sizes[ridx] - roffset);

        sbuf = (char *)temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *)temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        
        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        
        if (!sendnow && !recvnow)
        {
            
        }
        else if (!sendnow)
        { 
            mpi_errno =
                MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno)
            {
                exit(-1);
            }
            torecv -= recvnow;
        }
        else if (!recvnow)
        { 
            mpi_errno =
                MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno)
            {
                exit(-1);
            }
           
            tosend -= sendnow;
        }
        else
        {
            mpi_errno = MPI_Sendrecv(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG,
                                     rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG,
                                     comm, &status);
            if (mpi_errno)
            {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx])
        {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx])
        {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++)
    {
        hZCCL_float_decompress_openmp_threadblock_arg((char *)recvbuf + displs[i] * recvtype_extent, recvcounts[i], absErrBound, blockSize, (char *)temp_recvbuf + displs[i] * recvtype_extent);
    }
    free(temp_recvbuf);
    free(compressed_sizes);
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPI_Allreduce_hZCCL_RI2_mt_oa_record(const void *sendbuf,
                                       void *recvbuf,
                                       float compressionRatio,
                                       float tolerance,
                                       int blockSize,
                                       MPI_Aint count,
                                       MPI_Datatype datatype,
                                       MPI_Op op,
                                       MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS, mpi_errno_ret = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    size_t extent;
    MPI_Aint lb, true_extent;
    MPI_Aint *cnts, *displs; 
    int send_rank, recv_rank, total_count;
    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = count;
    void *tmpbuf;
    int tag;
    int flag;
    MPI_Request reqs[2]; 
    MPI_Status stas[2];
    extent = sizeof(datatype);
    is_inplace = (sendbuf == MPI_IN_PLACE);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

   
    size_t outSize;
    size_t byteLength;

    cnts = (MPI_Aint *)malloc(nranks * sizeof(MPI_Aint));
    assert(cnts != NULL);
    displs = (MPI_Aint *)malloc(nranks * sizeof(MPI_Aint));
    assert(displs != NULL);

    for (i = 0; i < nranks; i++)
        cnts[i] = 0;

    total_count = 0;
    for (i = 0; i < nranks; i++)
    {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count)
        {
            cnts[i] = count - total_count;
            break;
        }
        else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    
    if (!is_inplace)
    {
        memcpy(recvbuf, sendbuf, sizeof(datatype) * count);
    }
    

    tmpbuf = (void *)malloc(tmp_len * extent);
    
    unsigned char *outputBytes = (unsigned char *)malloc(tmp_len * extent);
    float *newData = (float *)malloc(cnts[0] * extent);

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++)
    {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

        
        tag = tag_base;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank] * extent, MPI_BYTE, src, tag, comm, &reqs[0]);
        if (mpi_errno)
        {
            exit(-1);
        }
        MPI_Test(&reqs[0], &flag, &stas[0]);
        hZCCL_float_openmp_threadblock_arg(outputBytes, (char *)recvbuf + displs[send_rank] * extent, &outSize, absErrBound, cnts[send_rank], blockSize);
        unsigned char *bytes = outputBytes;
        mpi_errno = MPI_Isend(bytes, outSize,
                              MPI_BYTE, dst, tag, comm, &reqs[1]);
        if (mpi_errno)
        {
            exit(-1);
        }
        if (mpi_errno)
        {
            exit(-1);
        }
        MPI_Wait(&reqs[0], &stas[0]);

        MPI_Test(&reqs[1], &flag, &stas[1]);
        hZCCL_float_decompress_openmp_threadblock_arg(newData, cnts[recv_rank], absErrBound, blockSize, tmpbuf);
        mpi_errno = MPI_Reduce_local(newData, (char *)recvbuf + displs[recv_rank] * extent,
                                     cnts[recv_rank], datatype, op);

        
        MPI_Wait(&reqs[1], &stas[1]);
        if (mpi_errno)
        {
            exit(-1);
        }
        
    }
    
    mpi_errno = MPIR_Allgatherv_intra_ring_RI2_mt_oa_record(MPI_IN_PLACE, -1, MPI_DATATYPE_NULL, recvbuf, cnts,
                                                            displs, datatype, comm, outputBytes, compressionRatio, tolerance,
                                                            blockSize);
    if (mpi_errno)
    {
        exit(-1);
    }

    free(outputBytes);
    free(cnts);
    free(displs);
    free(tmpbuf);
    free(newData);


    return mpi_errno;
}



int MPIR_Allgatherv_intra_ring_RI2_st_oa_record(const void *sendbuf,
                                                MPI_Aint sendcount,
                                                MPI_Datatype sendtype,
                                                void *recvbuf,
                                                const MPI_Aint *recvcounts,
                                                const MPI_Aint *displs,
                                                MPI_Datatype recvtype,
                                                MPI_Comm comm,
                                                unsigned char *outputBytes,
                                                float compressionRatio,
                                                float tolerance,
                                                int blockSize)
{
    int comm_size, rank, i, left, right;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;
    int total_count;
    recvtype_extent = sizeof(recvtype);
    double absErrBound = compressionRatio;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    
    size_t outSize;
    size_t byteLength;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    if (sendbuf != MPI_IN_PLACE)
    {
        
        memcpy((char *)recvbuf + displs[rank] * sizeof(recvtype),
               sendbuf, sizeof(data_type) * sendcount);
    }

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    MPI_Aint torecv, tosend, max, chunk_count;

    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;
    char *sbuf, *rbuf, *osbuf, *orbuf;

    int *compressed_sizes = (int *)malloc(comm_size * sizeof(int));
    
    void *temp_recvbuf = (void *)malloc(total_count * recvtype_extent);

   
    osbuf = (char *)recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
    sbuf = (char *)temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);

    hZCCL_float_single_thread_arg(sbuf, osbuf, &outSize, absErrBound,
                                     (recvcounts[sidx]), blockSize);
  

    int send_outSize = outSize;
    MPI_Allgather(&send_outSize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);


    
    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += compressed_sizes[i];
    if (total_count == 0)
        goto fn_exit;
    torecv = total_count - compressed_sizes[rank];
    tosend = total_count - compressed_sizes[right];

    chunk_count = 0;
    max = compressed_sizes[0];
    for (i = 1; i < comm_size; i++)
        if (max < compressed_sizes[i])
            max = compressed_sizes[i];
    if (MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE > 0 &&
        max > MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE)
    {
        chunk_count = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;
        
        if (!chunk_count)
            chunk_count = 1;
    }
    
    if (!chunk_count)
        chunk_count = max;

    while (tosend || torecv)
    { 
        MPI_Aint sendnow, recvnow;
        sendnow = ((compressed_sizes[sidx] - soffset) >
                   chunk_count)
                      ? chunk_count
                      : (compressed_sizes[sidx] - soffset);
        recvnow = ((compressed_sizes[ridx] - roffset) >
                   chunk_count)
                      ? chunk_count
                      : (compressed_sizes[ridx] - roffset);

        sbuf = (char *)temp_recvbuf + ((displs[sidx]) * recvtype_extent + soffset);
        rbuf = (char *)temp_recvbuf + ((displs[ridx]) * recvtype_extent + roffset);

        
        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

       
        if (!sendnow && !recvnow)
        {
            
        }
        else if (!sendnow)
        { 
            mpi_errno =
                MPI_Recv(rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG, comm, &status);
            if (mpi_errno)
            {
                exit(-1);
            }
            torecv -= recvnow;
        }
        else if (!recvnow)
        { 
            mpi_errno =
                MPI_Send(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG, comm);
            if (mpi_errno)
            {
                exit(-1);
            }
            tosend -= sendnow;
        }
        else
        { 
            mpi_errno = MPI_Sendrecv(sbuf, sendnow, MPI_BYTE, right, MPIR_ALLGATHERV_TAG,
                                     rbuf, recvnow, MPI_BYTE, left, MPIR_ALLGATHERV_TAG,
                                     comm, &status);
            if (mpi_errno)
            {
                exit(-1);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }
        soffset += sendnow;
        roffset += recvnow;
        if (soffset == compressed_sizes[sidx])
        {
            soffset = 0;
            sidx = (sidx + comm_size - 1) % comm_size;
        }
        if (roffset == compressed_sizes[ridx])
        {
            roffset = 0;
            ridx = (ridx + comm_size - 1) % comm_size;
        }
    }
    for (i = 0; i < comm_size; i++)
    {
        hZCCL_float_decompress_single_thread_arg((char *)recvbuf + displs[i] * recvtype_extent, recvcounts[i], absErrBound, blockSize, (char *)temp_recvbuf + displs[i] * recvtype_extent);
    }
    free(temp_recvbuf);
    free(compressed_sizes);
   
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


int MPI_Allreduce_hZCCL_RI2_st_oa_record(const void *sendbuf,
                                       void *recvbuf,
                                       float compressionRatio,
                                       float tolerance,
                                       int blockSize,
                                       MPI_Aint count,
                                       MPI_Datatype datatype,
                                       MPI_Op op,
                                       MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS, mpi_errno_ret = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    size_t extent;
    MPI_Aint lb, true_extent;
    MPI_Aint *cnts, *displs; 
    int send_rank, recv_rank, total_count;
    int tmp_len = 0;
    double absErrBound = compressionRatio;
    tmp_len = count;
    void *tmpbuf;
    int tag;
    int flag;
    MPI_Request reqs[2]; 
    MPI_Status stas[2];
    extent = sizeof(datatype);
    is_inplace = (sendbuf == MPI_IN_PLACE);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

   
    size_t outSize;
    size_t byteLength;

    cnts = (MPI_Aint *)malloc(nranks * sizeof(MPI_Aint));
    assert(cnts != NULL);
    displs = (MPI_Aint *)malloc(nranks * sizeof(MPI_Aint));
    assert(displs != NULL);

    for (i = 0; i < nranks; i++)
        cnts[i] = 0;

    total_count = 0;
    for (i = 0; i < nranks; i++)
    {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count)
        {
            cnts[i] = count - total_count;
            break;
        }
        else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

   
    if (!is_inplace)
    {
        memcpy(recvbuf, sendbuf, sizeof(datatype) * count);
    }
    

    tmpbuf = (void *)malloc(tmp_len * extent);
 
    unsigned char *outputBytes = (unsigned char *)malloc(tmp_len * extent);
    float *newData = (float *)malloc(cnts[0] * extent);

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++)
    {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

     
        tag = tag_base;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank] * extent, MPI_BYTE, src, tag, comm, &reqs[0]);
        if (mpi_errno)
        {
            exit(-1);
        }
        MPI_Test(&reqs[0], &flag, &stas[0]);
        hZCCL_float_single_thread_arg(outputBytes, (char *)recvbuf + displs[send_rank] * extent, &outSize, absErrBound, cnts[send_rank], blockSize);
        unsigned char *bytes = outputBytes;
        mpi_errno = MPI_Isend(bytes, outSize,
                              MPI_BYTE, dst, tag, comm, &reqs[1]);
        if (mpi_errno)
        {
            exit(-1);
        }
        if (mpi_errno)
        {
            exit(-1);
        }
        MPI_Wait(&reqs[0], &stas[0]);

        MPI_Test(&reqs[1], &flag, &stas[1]);
        hZCCL_float_decompress_single_thread_arg(newData, cnts[recv_rank], absErrBound, blockSize, tmpbuf);
        mpi_errno = MPI_Reduce_local(newData, (char *)recvbuf + displs[recv_rank] * extent,
                                     cnts[recv_rank], datatype, op);

      
        MPI_Wait(&reqs[1], &stas[1]);
        if (mpi_errno)
        {
            exit(-1);
        }
    }
    
    mpi_errno = MPIR_Allgatherv_intra_ring_RI2_st_oa_record(MPI_IN_PLACE, -1, MPI_DATATYPE_NULL, recvbuf, cnts,
                                                            displs, datatype, comm, outputBytes, compressionRatio, tolerance,
                                                            blockSize);
    if (mpi_errno)
    {
        exit(-1);
    }

    free(outputBytes);
    free(cnts);
    free(displs);
    free(tmpbuf);
    free(newData);

    return mpi_errno;
}
