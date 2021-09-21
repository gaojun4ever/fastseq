from model_helper import GPT2Helper
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model_name_or_path',
                    required=True,
                    type=str,
                    help='Model path, or pretrained model name')

parser.add_argument('--cache_dir',
                    required=False,
                    type=str,
                    default=os.path.join('.', 'cache_models'),
                    help='Directory to cache pre-trained models')

parser.add_argument('--onnx_model_path',
                    required=True,
                    type=str,
                    help='Path to onnx models')

parser.add_argument('--prompt_text',
                    required=False,
                    type=str,
                    default="here is an example of gpt2 model",
                    help='input text')
sampling_option_group = parser.add_argument_group("one step sampling options")

sampling_option_group.add_argument('--do_sample',
                                    action='store_true',
                                    help='If to do sampling instead of beam search or greedy.')
sampling_option_group.add_argument('--do_sample_top_p',
                                    type=float,
                                    default=0.95,
                                    help='Nuclear/top-p sampling accumulation probability.')
sampling_option_group.add_argument('--do_sample_top_k', type=int, default=5, help='Use top-k if non-zero.')
args = parser.parse_args()

config, tokenizer = GPT2Helper.get_config_and_tokenizer(args.model_name_or_path, args.cache_dir)

ort_session = GPT2Helper.create_onnxruntime_session(args.onnx_model_path)

output = GPT2Helper.top_k_top_p_decoding(input_text=args.prompt_text, 
                                config=config, 
                                tokenizer=tokenizer, 
                                ort_session=ort_session, 
                                top_k=args.do_sample_top_k,
                                max_decoding_length=50)


print(output)
