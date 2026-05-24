# GPTQ

项目基于原版gptq仓库

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)

先用download_model中的程序下载模型

实验命令样例：
CUDA_VISIBLE_DEVICES=1 python opt2.py models/opt-2.7b c4 --wbits 2