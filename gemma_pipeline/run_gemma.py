import onnxruntime as ort
import os
import numpy as np
import time
import gc
import psutil
import string

from pathlib import Path
from tokenizers import Tokenizer

QUESTION_TOKENS = ["?", "Why", "How", "What", "Could", "Would", "Should", "Can"]

def apply_question_penalty(logits, tokenizer, penalty=2.0):
    for token in QUESTION_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            logits[token_id] /= penalty
    return logits

def softmax_numpy(x: np.array, temperature: float=1) -> np.array:
    # stabilize x in case of large numbers 
    x = x - np.max(x, axis=-1, keepdims=True)

    # Apply temperature
    x = x/temperature

    # Apply Softmax
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def top_k_probas(probas: np.array, k: int=5) -> np.array:
    # Copy probas so in-place operations don't work on original variable
    probas = probas.copy()
    # Normalize probabilities
    probas /= np.sum(probas)
    # Using -probas to get in descending order
    top_indices_sorted = np.argsort(-probas)[:k]
    top_k_probas = probas[top_indices_sorted]

    # Renormalize top-k probabilites to sum to 1 (probabilites must sum to 1 to use np.random.choice
    top_k_probas /= np.sum(top_k_probas)

    # Return top k probabilities
    return top_indices_sorted, top_k_probas

def apply_repetition_penalty(logits, generated_ids, penalty=1.1):
    for token_id in set(generated_ids):
        logits[token_id] /= penalty
    return logits

class GemmaRunner:

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.onnx_root = Path(ort.__file__).parent

        # Subdirectory where all .onnx dependencies are located
        self.model_subdirectory = "gemma-3-1b-it-ONNX-GQA"

        # The embeddings model is entry point, use netron to visualize
        self.model_name = "model_q4f16.onnx"

        # Tokenizer
        self.tokenizer_json = "tokenizer.json"


        self.model_path = self.root_dir/"models"/self.model_subdirectory/self.model_name
        self.tokenizer_path = self.root_dir/"models"/self.model_subdirectory/self.tokenizer_json
        self.hexagon_driver = self.onnx_root/"capi"/"QnnHtp.dll"


    def create_session(self, init_query="<start_of_turn>user\nWhy is the sky blue?<end_of_turn><start_of_turn>model"):
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        
        # Creating an inference session for the embedding graph
        model_session = ort.InferenceSession(self.model_path,
                                        # providers= [("QNNExecutionProvider",qnn_provider_options)],
                                    sess_options= session_options
                              )
        
        inputs = model_session.get_inputs()
        outputs = model_session.get_outputs()

        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        encoding = tokenizer.encode(init_query)
        input_ids = encoding.ids

        batch_size = 1
        num_kv_heads = 1
        attn_head_size = 256
        num_layers = 26
        past_seq_length = len(input_ids)

        empty_kv = {}
        for i in range(num_layers):
            # Shape of key and value tensors for each transformer layer
            past_shape = (batch_size, num_kv_heads, past_seq_length, attn_head_size)

            # Initialize past keys for layer i (used in attention mechanism to avoid recomputation
            empty_kv[f"past_key_values.{i}.key"] = np.zeros(past_shape, dtype=np.float32)

            # Initialize past values for layer i
            empty_kv[f"past_key_values.{i}.value"] = np.zeros(past_shape, dtype=np.float32)


        attention_mask = np.ones((batch_size, past_seq_length), dtype=np.int64)
        
        input_ids = np.array([input_ids], dtype=np.int64)
        position_ids = np.arange(past_seq_length, dtype=np.int64).reshape(1, -1)

        input_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            **empty_kv
        }

        session_output = model_session.run(None, input_dict)

        batch_size = 1
        hidden_size = 1152
        num_heads = 4
        attn_head_size = 256 
        num_layers = 26
        max_seq_len = 512
        num_key_value_heads = 1

        present_kv = {f"past_key_values.{i}.key": session_output[1 + i * 2] for i in range(num_layers)}
        present_kv.update({f"past_key_values.{i}.value": session_output[1 + i * 2 + 1] for i in range(num_layers)})
        
        logits = session_output[0]


        softmax = lambda x, temperature=1: np.exp((x-np.max(x, axis=-1, keepdims=True))/temperature)/np.sum(np.exp((x-np.max(x, axis=-1, keepdims=True))/temperature), axis=-1, keepdims=True)
        temp = 0.6
        probas = softmax(logits[0,-1], temperature=temp)
        # probas = probas / probas.sum()
        next_token_id = int(np.random.choice(len(probas), p=probas)) #int(np.argmax(probas))

        tokenizer.decode([next_token_id])

        temp = 0.3
        start = time.time()
        max_tokens = 1000
        top_k = 20
        generated_ids = [next_token_id]
        prev_seq_len = logits.shape[1]
        printed_length = 0
        # This is the next position (131)
        position_itr = input_ids.shape[-1]
        # position_ids = np.array((batch_size, 1), dtype=np.int64)
        # print(prev_seq_len)
        # print(attention_mask.shape)
        print("\nInitial Query:\n", init_query)
        for _ in range(max_tokens):

            input_ids = np.array([[next_token_id]], dtype=np.int64)
            position_ids = np.array([[position_itr]], dtype=np.int64)
            # print(tokenizer.decode(generated_ids, skip_special_tokens=True))
            # print(tokenizer.decode([next_token_id], skip_special_tokens=False),end=" ")
            iter_inputs = {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "position_ids": position_ids,
            **present_kv,
            }

            session_output = model_session.run(None, iter_inputs)
            prev_seq_len += 1
            
            # Update attention mask
            attention_mask = np.ones((batch_size, prev_seq_len), dtype=np.int64)
            
            # Update position id
            position_itr += 1
            
            # Update KV Cache
            present_kv = {f"past_key_values.{i}.key": session_output[1 + i * 2] for i in range(num_layers)}
            present_kv.update({f"past_key_values.{i}.value": session_output[1 + i * 2 + 1] for i in range(num_layers)})
            # print(prev_seq_len)
            # print(present_kv.get("past_key_values.0.key").shape)
            # print(len(attention_mask))
            logits = session_output[0]

            token_logits = logits[0,-1]
            token_logits = apply_repetition_penalty(token_logits, generated_ids, penalty=1.1)
            token_logits = apply_repetition_penalty(token_logits, generated_ids, penalty=1.1)
            token_logits = apply_question_penalty(token_logits, tokenizer, penalty=2.0)
            # Get probabilities
            probas = softmax(token_logits, temperature=temp)
            top_indices, top_probas = top_k_probas(probas, k=top_k) 
            next_token_id = int(np.random.choice(top_indices, p=top_probas)) #int(np.argmax(probas))
            generated_ids.append(next_token_id)
            
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = full_text[printed_length:]
            if new_text:
                printed_length = len(full_text)



            if next_token_id == 106:
                break

            # print(new_text, end="")

            # # Stop if any character is not in the allowed set
            # if any(c not in allowed_chars for c in decoded_token):
            #     break
        
        end = time.time()
        elapsed = end - start
        tps = np.round((max_tokens / elapsed), 2)
        print(f"\nTokens Per Second: {tps}")
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text

# if __name__ == "__main__":
#     runner = GemmaRunner()
#     runner.create_session(
#         init_query="<start_of_turn>user\nGive a SHORT summary of the following text: Ok, so for this weekend's homework, we're going to read Anna Karenina chapters 1-6 and write a 1000 word analysis of the relationship between Anna and Levin<end_of_turn><start_of_turn>model\n"
#     )