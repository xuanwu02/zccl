# ZCCL: The Compression-accelerated Collective Communications Library

* Developer: Jiajun Huang (Github: JiajunHuang1999)

This is the official repo of ZCCL, The Compression-accelerated Collective Communications Library. ZCCL is based on the following research works.

* hZCCL [SC '24] https://dl.acm.org/doi/10.1109/SC41406.2024.00110
* ZCCL [IPDPS '24] https://www.computer.org/csdl/proceedings-article/ipdps/2024/871100a752/1YpzZT6urCg

## About ZCCL

ZCCL is a *first-ever* compression-accelerated collective communications library, which can directly communicate and compute with compressed data.

## Citation

If you find ZCCL useful in your research or applications, we kindly invite you to cite our papers. Your support helps advance the field and acknowledges the contributions of our work. Thank you!

- **[IPDPS '24]** An Optimized Error-controlled MPI Collective Framework Integrated with Lossy Compression
    ```bibtex
    @inproceedings{huang2023ccoll,
        title={An Optimized Error-controlled MPI Collective Framework Integrated with Lossy Compression},
        author = {Huang, Jiajun and Di, Sheng and Yu, Xiaodong and Zhai, Yujia and Liu, Jinyang and Jian, Zizhe and Liang, Xin and Zhao, Kai and Lu, Xiaoyi and Chen, Zizhong and Cappello, Franck and Guo, Yanfei and Thakur, Rajeev},
        booktitle = {Proceedings of the 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
        year = {2024}
    }
    ```

- **[SC '24]** hZCCL: Accelerating Collective Communication with Co-Designed Homomorphic Compression
    ```bibtex
    @inproceedings{huang2024hZCCL,
        title={hZCCL: Accelerating Collective Communication with Co-Designed Homomorphic Compression},
        author = {Huang, Jiajun and Di, Sheng and Yu, Xiaodong and Zhai, Yujia and Zhang, Zhaorui and Liu, Jinyang and Lu, Xiaoyi and Raffenetti, Ken and Zhou, Hui and Zhao, Kai and Chen, Zizhong and Cappello, Franck and Guo, Yanfei and Thakur, Rajeev},
        booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis},
        year = {2024}
    }
    ```

## Installation
Configure and build ZCCL:
```bash
# Decompress the downloaded files
# Change to ZCCL directory and set up the installation directory:
cd ZCCL
mkdir install

# Run the configuration script:
./autogen.sh
./configure --prefix=$(pwd)/install --enable-openmp CC=mpicc

# Compile ZCCL using multiple threads:
make -j

# Install the compiled ZCCL:
make install

# Change to the jobs directory to proceed with running the ZCCL:
cd jobs
```
