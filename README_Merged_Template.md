---
base_model: "Qwen/Qwen3-4B-Instruct-2507"
datasets:
- {dataset_id}
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- merged
- structured-output
---

qwen3-4b-structured-sft-merged

This repository provides a **merged model** based on
**Qwen3-4B-Instruct-2507** fine-tuned using **QLoRA (4-bit, Unsloth)**.

This repository contains **merged model weights**.

## Training Objective

This model is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.

## Training Configuration

- Base model: Qwen3-4B-Instruct-2507
- Method: QLoRA (4-bit)
- Max sequence length: {max_seq_len}
- Epochs: {epochs}
- Learning rate: {lr_str}
- LoRA: r={lora_r}, alpha={lora_alpha}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{hf_lora_repo}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

## Sources & Terms (IMPORTANT)

Training data: {dataset_id}

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
