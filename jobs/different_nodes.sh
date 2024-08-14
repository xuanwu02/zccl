#!/bin/bash

#SBATCH --job-name=different_nodes
#SBATCH --account=$XXX
#SBATCH --partition=$XXX
#SBATCH --nodes=512
#SBATCH --output=different_nodes.txt
#SBATCH --error=different_nodes.error
#SBATCH --mem-per-cpu=100gb
#SBATCH --time=01:00:00

# Setup My Environment
export PATH=mpich/install/bin:$PATH


# MPI setup
NNODES=512
NRANKS_PER_NODE=1
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
#Compression setup
DATADIR=$XXX
ERRORBOUND=1E-4
BLOCKSIZE=36
KERNELMAX=4
KERNELMIN=0
# START_SIZE=52428800

# export OMP_NUM_THREADS=$NUMTHREADS




NROUNDS=1
START_SCALE=2
# Run My Program
cd $XXX/hZCC-artifact/install/bin
echo
echo -e "Running compression-accelerated allreduce with different node counts\n"

echo -e "NNODES: $NNODES, BLOCKSIZE: $BLOCKSIZE, ERRORBOUND: $ERRORBOUND, KERNELMAX: $KERNELMAX, KERNELMIN: $KERNELMIN\n"

let "NUMTHREADS=18"

let "SIZEMAX=$XXX"


echo -e "SIZEMAX = $SIZEMAX, DATADIR: $DATADIR, NUMTHREADS: $NUMTHREADS\n"

let "SCALE=$START_SCALE"
let "kernel=$KERNELMIN"
let "int=1"
let "counter=1"
export OMP_NUM_THREADS=$NUMTHREADS
while(( $int<=$NROUNDS ))
do
    while(( $kernel<=${KERNELMAX} ))
    do
        echo
        echo -e "Kernel $kernel ($START_SCALE,${NNODES},x2)\n"
        while(( $SCALE<=${NNODES} ))
        do
            echo "SCALE = ${SCALE}"
            echo 
            mpiexec -n $SCALE --ppn 1 -bind-to socket:1 ./compression_allreduce -v 0 -s $SIZEMAX -l $SIZEMAX -i 10 -w 0 -k $kernel -f $DATADIR -r $ERRORBOUND -b $BLOCKSIZE -a 0
            let "SCALE=SCALE*2"
            if (( $counter>1 ))
            then
                let "counter=1"
                break
            fi
            if (( $SCALE>=${NNODES} ))
            then
                let "SCALE=${NNODES}"
                let "counter=counter+1"
            fi
        done
        let "SCALE=$START_SCALE"
        let "kernel=kernel+1"
    done
    let "SCALE=$START_SCALE"
    let "kernel=$KERNELMIN"
    let "int=int+1"
done