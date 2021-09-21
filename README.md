# Fastseq
基于ONNXRUNTIME的文本生成加速框架

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
# 将huggingface保存的 模型/checkpoint 转换为onnx格式。这里使用onnxruntime自带的转换工具。
python -m onnxruntime.transformers.convert_to_onnx \
    -m "path_to_checkpoint/model_name(gpt2)" \
    --model_class GPT2LMHeadModel \
    --output gpt2_fp32.onnx \
    -p fp32
```

### 3. DEMO测试
```shell
CUDA_VISIBLE_DEVICES=3 python demo.py \
    --onnx_model_path "./gpt2_fp32.onnx" \
    --model_name_or_path "path_to_checkpoint" \
    --prompt_text "here is an example of gpt2 model" \
    --do_sample_top_k 5
```


### 4. TODO
- [x] TopK Decoding
- [ ] Beam Search Decoding
- [ ] TensorRT Provider
- [ ] ONNXRUNTIME Framework for BERT 
- [ ] Benchmark test
- [ ] RESTful server demo
- [ ] ONNXRUNTIME 1.8.1 Support


