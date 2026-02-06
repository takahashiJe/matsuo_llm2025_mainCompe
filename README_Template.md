base_model: "Qwen/Qwen3-4B-Instruct-2507"
datasets:
- {dataset_id}
language:
- en
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- lora
- structured-output
---

qwen3-4b-structured-sft-lora

This repository provides a **LoRA adapter** fine-tuned from
**Qwen/Qwen3-4B-Instruct-2507** using **QLoRA (4-bit, Unsloth)**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.

## Training Objective

This adapter is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.

## Training Configuration

- Base model: Qwen/Qwen3-4B-Instruct-2507
- Method: QLoRA (4-bit)
- Max sequence length: {max_seq_len}
- Epochs: {epochs}
- Learning rate: {lr_str}
- LoRA: r={lora_r}, alpha={lora_alpha}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "JuntaTakahashi/qwen3-4b-structured-sft-lora"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & Terms (IMPORTANT)

Training data: {dataset_id}

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
"""