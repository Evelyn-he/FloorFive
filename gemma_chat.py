from transformers import AutoConfig, AutoTokenizer
import onnxruntime
import numpy as np
import os
import subprocess
import urllib.request
# --------------------------
# 0. Clone repo if missing
# --------------------------

_PATH_TO_MODEL = os.environ.get("GEMMA_ONNX_PATH", "gemma-3-1b-it-ONNX")
_ONNX_DIR = os.path.join(_PATH_TO_MODEL, "onnx")
os.makedirs(_ONNX_DIR, exist_ok=True)

# URLs for raw files (not the GitHub-style blob links)
FILE_URLS = {
    "model.onnx":      "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model.onnx",
    "model.onnx_data": "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model.onnx_data",
}

for filename, url in FILE_URLS.items():
    target_path = os.path.join(_ONNX_DIR, filename)
    if not os.path.exists(target_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, target_path)
    else:
        print(f"{filename} already exists, skipping.")


# 1. Load config, processor, and model
path_to_model = "gemma-3-1b-it-ONNX"
config = AutoConfig.from_pretrained(path_to_model)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
decoder_session = onnxruntime.InferenceSession(f"{path_to_model}/onnx/model.onnx")

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers

def generate_response(prompt, max_new_tokens=1024):
    eos_token_id = max_new_tokens # 106 is for <end_of_turn>
    messages = [
    { "role": "system", "content": "You are a really helpful assistant." },
    { "role": "user", "content": prompt },
    ]

    _eos_token_id = tokenizer.eos_token_id

    ## Apply tokenizer
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")

    ## Prepare decoder inputs
    batch_size = inputs['input_ids'].shape[0]
    past_key_values = {
        f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
        for layer in range(num_hidden_layers)
        for kv in ('key', 'value')
    }
    input_ids = inputs['input_ids'].astype(np.int64)
    position_ids = np.tile(np.arange(1, input_ids.shape[-1] + 1), (batch_size, 1)).astype(np.int64)

    # 3. Generation loop
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)
    for i in range(max_new_tokens):
        ort_inputs = {
            "input_ids":    input_ids,
            "position_ids": position_ids,
            **past_key_values
        }
        logits, *present_values = decoder_session.run(None, ort_inputs)

        next_token = logits[:, -1].argmax(-1, keepdims=True)
        generated_tokens = np.concatenate([generated_tokens, next_token], axis=-1)

        next_word = tokenizer.decode(next_token[0])
        if next_word == "<end_of_turn>":
            break
        # print(next_word, end="", flush=True)

        if next_token.item() == _eos_token_id:
            break

        input_ids   = next_token
        position_ids = position_ids[:, -1:] + 1
        for idx, key in enumerate(past_key_values):
            past_key_values[key] = present_values[idx]

    response = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
    return response