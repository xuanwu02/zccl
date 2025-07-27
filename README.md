# ZCCL: The Compression-accelerated Collective Communications Library

* Developer and maintainer: ZCCL.org ([Team Members](https://zccl.org/team/))
* Contact: zcclorg@gmail.com

This is the official repo of ZCCL, The Compression-accelerated Collective Communications Library. ZCCL is based on the following research works.

- **hZCCL**:
[SC '24] https://dl.acm.org/doi/10.1109/SC41406.2024.00110
- **ZCCL**: 
[IPDPS '24] https://www.computer.org/csdl/proceedings-article/ipdps/2024/871100a752/1YpzZT6urCg 
[Journal Extension] https://arxiv.org/abs/2502.18554



## About ZCCL

ZCCL is a *first-ever* compression-accelerated collective communications library, which can directly communicate and compute with compressed data.

## Citation

If you find ZCCL useful in your research or applications, we kindly invite you to cite our papers. Your support helps advance the field and acknowledges the contributions of our work. Thank you!

- **[IPDPS '24]** An Optimized Error-controlled MPI Collective Framework Integrated with Lossy Compression
    ```bibtex
    @inproceedings{huang2023ccoll,
        title={An Optimized Error-controlled MPI Collective Framework Integrated with Lossy Compression},
        author = {Huang, Jiajun and Di, Sheng and Yu, Xiaodong and Zhai, Yujia and Zhang, Zhaorui and Liu, Jinyang and Lu, Xiaoyi and Raffenetti, Ken and Zhou, Hui and Zhao, Kai and Chen, Zizhong and Cappello, Franck and Guo, Yanfei and Thakur, Rajeev},
        booktitle = {Proceedings of the 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
        year = {2024}
    }
    ```

- **[Journal Extension]** ZCCL: Significantly Improving Collective Communication With Error-Bounded Lossy Compression
    ```bibtex
    @misc{huang2024ZCCL,
        title={ZCCL: Significantly Improving Collective Communication With Error-Bounded Lossy Compression},
        author = {Huang, Jiajun and Di, Sheng and Yu, Xiaodong and Zhai, Yujia and Zhang, Zhaorui and Liu, Jinyang and Lu, Xiaoyi and Raffenetti, Ken and Zhou, Hui and Zhao, Kai and Alharthi, Khalid, and Chen, Zizhong and Cappello, Franck and Guo, Yanfei and Thakur, Rajeev},
        url={https://arxiv.org/abs/2502.18554},
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
### Installation Note
1. if seeing `configure.ac:22: error possibly undefinded macro: AC_PROG_LIBTOOL` when running command `./autogen.sh`, simply replace 'AC_PROG_LIBTOOL' with 'LT_INIT' at line 22 on 'configure.ac'.
2. Some prerequisite packages needed in Linux system
    - 'sudo apt update && sudo apt-get install libtool automake autoconf mpich'
3. You may encounter broken MPI environment setup in some virtual environments like WSL2, please consider installing 'openmpi-bin' and 'libopenmpi-dev' then. OpenMPI is often more robust in virtual desktop and WSL environments.