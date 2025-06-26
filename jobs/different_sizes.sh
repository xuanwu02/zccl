#!/bin/bash

#SBATCH --job-name=different_sizes
#SBATCH --account=$XXX
#SBATCH --partition=$XXX
#SBATCH --nodes=64
#SBATCH --output=different_sizes.txt
#SBATCH --error=different_sizes.error
#SBATCH --mem-per-cpu=100gb
#SBATCH --time=00:30:00

# Setup My Environment
export PATH=mpich/install/bin:$PATH


# MPI setup
NNODES=64
NRANKS_PER_NODE=1
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
#Compression setup
DATADIR=$XXX
ERRORBOUND=1E-4
BLOCKSIZE=36
KERNELMAX=4
KERNELMIN=0
START_SIZE=52428800
NUMTHREADS=16

NROUNDS=1
# Run My Program
cd $XXX/ZCCL/install/bin

echo -e "Running compression-accelerated allreduce with different data sizes\n"

echo -e "NNODES: $NNODES, DATADIR: $DATADIR, BLOCKSIZE: $BLOCKSIZE, ERRORBOUND: $ERRORBOUND, KERNELMAX: $KERNELMAX, KERNELMIN: $KERNELMIN, START_SIZE: $START_SIZE, NUMTHREADS: $NUMTHREADS\n"

export OMP_NUM_THREADS=$NUMTHREADS
let "int=1"
let "kernel=$KERNELMIN"
let "SIZE=$START_SIZE"
let "counter=1"
while(( $int<=$NROUNDS ))
do
    while(( $kernel<=${KERNELMAX} ))
    do
        echo -e "Kernel $kernel (52428800,$XXX,52428800 or 50MB)\n"
        while(( $SIZE<=$XXX ))
        do
            mpiexec -n $NNODES -ppn 1 -bind-to socket:1 ./compression_allreduce -v 0 -s $SIZE -l $SIZE -i 10 -w 0 -k $kernel -f $DATADIR -r $ERRORBOUND -b $BLOCKSIZE
            let "SIZE=SIZE+52428800"
            if (( $counter>1 ))
            then
                let "counter=1"
                break
            fi
            if (( $SIZE>=$XXX ))
            then
                let "SIZE=$XXX"
                let "counter=counter+1"
            fi
        done
        let "SIZE=$START_SIZE"
        let "kernel=kernel+1"
    done
let "SIZE=$START_SIZE"
let "kernel=$KERNELMIN"
let "int=int+1"
done
