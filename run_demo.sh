CUDA_VISIBLE_DEVICES=3 python demo.py \
    --onnx_model_path "./gpt2_fp32.onnx" \
    --model_name_or_path "gpt2" \
    --do_sample_top_k 5