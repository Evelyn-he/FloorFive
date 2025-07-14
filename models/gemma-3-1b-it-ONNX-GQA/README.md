---
pipeline_tag: text-generation
base_model:
- google/gemma-3-1b-it
library_name: transformers.js
license: gemma
new_version: onnx-community/gemma-3-1b-it-ONNX-GQA
---

## Usage
# Inference using Snapdragon X Elite

### ONNXRuntime

```py
from transformers import AutoConfig, AutoTokenizer
import onnxruntime
import numpy as np

# 1. Load config, processor, and model
path_to_model = "./gemma-3-1b-it-ONNX"
config = AutoConfig.from_pretrained(path_to_model)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
decoder_session = onnxruntime.InferenceSession(f"{path_to_model}/onnx/model.onnx")

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers
eos_token_id = 106 # 106 is for <end_of_turn>

# 2. Prepare inputs
## Create input messages
messages = [
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Write me a poem about Machine Learning." },
]

## Apply tokenizer
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
input_ids = inputs['input_ids']
position_ids = np.tile(np.arange(1, input_ids.shape[-1] + 1), (batch_size, 1))

# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
for i in range(max_new_tokens):
  logits, *present_key_values = decoder_session.run(None, dict(
      input_ids=input_ids,
      position_ids=position_ids,
      **past_key_values,
  ))

  ## Update values for next generation loop
  input_ids = logits[:, -1].argmax(-1, keepdims=True)
  position_ids = position_ids[:, -1:] + 1
  for j, key in enumerate(past_key_values):
    past_key_values[key] = present_key_values[j]

  generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
  if (input_ids == eos_token_id).all():
    break

  ## (Optional) Streaming
  print(tokenizer.decode(input_ids[0]), end='', flush=True)
print()

# 4. Output result
print(tokenizer.batch_decode(generated_tokens))
```

<details>
<summary>See example output</summary>

```
Okay, here’s a poem about Machine Learning, aiming for a balance of technical and evocative language:

**The Silent Learner**

The data streams, a boundless flow,
A river vast, where patterns grow.
No human hand to guide the way,
Just algorithms, come what may.

Machine Learning, a subtle art,
To teach a system, a brand new start.
With weights and biases, finely tuned,
It seeks the truth, beneath the moon.

It learns from errors, big and small,
Adjusting swiftly, standing tall.
From pixels bright to voices clear,
It builds a model, banishing fear.

Of blind prediction, cold and stark,
It finds the meaning, leaves its mark.
A network deep, a complex grace,
Discovering insights, time and space.

It sees the trends, the subtle hue,
Predicting futures, fresh and new.
A silent learner, ever keen,
A digital mind, unseen, serene.

So let the code begin to gleam,
A blossoming of a learning dream. 
Machine Learning, a wondrous sight,
Shaping the future, shining bright. 

---

Would you like me to:

*   Adjust the tone or style? (e.g., more technical, more metaphorical)
*   Focus on a specific aspect of ML (e.g., neural networks, data analysis)?
*   Create a different length or format?
```

</details>



### Transformers.js
```js
import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "onnx-community/gemma-3-1b-it-ONNX",
  { dtype: "q4" },
);

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Write me a poem about Machine Learning." },
];

// Generate a response
const output = await generator(messages, { max_new_tokens: 512, do_sample: false });
console.log(output[0].generated_text.at(-1).content);
```