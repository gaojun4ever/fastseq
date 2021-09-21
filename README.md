# Fastseq
基于ONNXRUNTIME的加速demo

### 1. 环境配置
```shell
# 创建onnx conda环境
conda create -n onnx_py38 python=3.8
conda activate onnx_py38
conda install pytorch cudatoolkit=10.2 -c pytorch

# 安装onnxruntime-gpu(目前只有1.5.2版本测试成功)
pip install onnxruntime-gpu==1.5.2

# 安装transformers==3.1.0版本
pip install transformers==3.1.0
```

### 2. ONNX转换
```shell
# 降huggingface保存的 模型/checkpoint 转换为onnx格式。这里使用onnxruntime自带的转换工具。

python -m onnxruntime.transformers.convert_to_onnx \
    -m "path_to_checkpoint/model_name(gpt2) \
    --model_class GPT2LMHeadModel \
    --output gpt2_fp32.onnx \
    -p fp32


```


