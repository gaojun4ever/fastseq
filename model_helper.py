from transformers import AutoTokenizer
from onnxruntime.transformers.gpt2_helper import Gpt2Helper
from transformers import AutoConfig
from transformers import GPT2LMHeadModel
import onnxruntime
import numpy
import torch
import time
import torch.nn.functional as F
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
# device = torch.device("cpu")


EXAMPLE_Text = ['here is an example of gpt2 model']


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    assert logits.dim() == 1  # batch size 1 for now
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
def get_example_inputs(prompt_text, config, tokenizer):
    num_attention_heads = config.n_head
    hidden_size = config.n_embd
    num_layer = config.n_layer
        
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)

    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
    
    return input_ids, attention_mask, position_ids, empty_past


def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                past_sequence_length=past[0].size(3),
                                                sequence_length=input_ids.size(1),
                                                config=config)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
    return outputs

class GPT2Helper:
    @staticmethod
    def get_config_and_tokenizer(model_name_or_path, cache_dir):
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return config, tokenizer
    
    @staticmethod
    def create_onnxruntime_session(onnx_model_path):
        session = onnxruntime.InferenceSession(onnx_model_path)
        return session
    
    
    
    @staticmethod
    def top_k_top_p_decoding(input_text, config, tokenizer, ort_session, 
                             top_k=5, top_p=0.0, max_decoding_length=30):
        input_text = [input_text]
        num_attention_heads = config.n_head
        hidden_size = config.n_embd
        num_layer = config.n_layer
        eos_token_id = torch.tensor(tokenizer.eos_token_id).to(device)
        
        input_ids, attention_mask, position_ids, past = get_example_inputs(input_text, config, tokenizer)
        batch_size = input_ids.size(0)

        has_eos = torch.zeros(batch_size, dtype=torch.bool).to(device)

        all_token_ids = input_ids.clone().to(device)
        for step in range(max_decoding_length):
            outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)

            next_token_logits = outputs[0][:, -1, :].squeeze()
            next_token_logits = top_k_top_p_filtering(logits=next_token_logits, top_k=5)
            # print(torch.multinomial(topk_logits,1))
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, 1)
            
            has_eos = has_eos | (next_tokens.item() == eos_token_id.item())
            tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
            all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            # Update input_ids, attention_mask, position_ids and past
            input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)    
            position_ids = (position_ids[:,-1] + 1).reshape(batch_size,1)
            attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)    

            past = []
            for i in range(num_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()
                past.append(past_i.to(device))

            if torch.all(has_eos):
                break
        tokenized_outputs = []
        for i, output in enumerate(all_token_ids):
            tokenized_outputs.append(tokenizer.decode(output, skip_special_tokens=True))
        return tokenized_outputs