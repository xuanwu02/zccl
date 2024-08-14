# hZCCL-artifact

* Developer: Jiajun Huang (Github: JiajunHuang1999)

This is the artifact of hZCCL, which has been accepted by SC '24.

## About hZCCL

hZCCL is a *first-ever* homomorphic compression-communication co-design, which enables operations to be performed directly on compressed data, saving the cost of time-consuming decompression and recompression. 

<details>
<summary>
Paper's Main Contributions
</summary>

- `C_1` We present a homomorphic compressor with a run-time heuristic to dynamically select efficient compression pipelines for reducing the cost of decompression-operation-compression (DOC) handling.
- `C_2` We present a homomorphic compression-communication co-design, hZCCL, which enables operations to be performed directly on compressed data, saving the cost of time-consuming decompression and recompression.
- `C_3` We evaluate hZCCL with up to 512 nodes and across five application datasets. The experimental results demonstrate that our homomorphic compressor achieves a single-socket CPU throughput of up to 379.08 GB/s, surpassing the conventional DOC workflow by up to 36.53X.
- `C_4` Moreover, our hZCCL-accelerated collectives outperform two state-of-the-art baselines, delivering speedups of up to 2.12X and 6.77X compared to original MPI collectives in single-thread and multi-thread modes, respectively, while maintaining the data accuracy.
</details>

## Installation
Configure and build the hZCCL-artifact:
```bash
# Decompress the downloaded files
# Change to the artifact directory and set up the installation directory:
cd hZCCL-artifact
mkdir install

# Run the configuration script:
./autogen.sh
./configure --prefix=$(pwd)/install --enable-openmp CC=mpicc

# Compile the artifact using multiple threads:
make -j

# Install the compiled artifact:
make install

# Change to the jobs directory to proceed with running the artifact:
cd jobs
```