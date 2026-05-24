# GPTQ

项目基于原版gptq仓库

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)
## 实验方法
先用download_model中的程序下载模型

实验命令样例：
CUDA_VISIBLE_DEVICES=0 nohup python llama.py models/llama-2-7b-hf wikitext2 --wbits 2 --groupsize 64 --stage1_hessian >log_2stage_64/wikitext/llama-2-7b/2bit/r0.log 2>&1 &

#### 可调方法与参数
在llama.py的开头调整要用的量化策略：  
  from gptq_2stage import *可改成：  
  gptq_2stage_2_multiround_EMA  
  gptq_2stage_3_multiround_flip  

用gptq_2stage_2_multiround_EMA时在该文件开头调整EMA轮数和参数alpha大小
