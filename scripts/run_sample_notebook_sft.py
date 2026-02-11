#!/usr/bin/env python
"""
Run SFT training equivalent to sample_notebooks/2025最終課題メインコンペ_標準コード1（SFT）.ipynb
as a standalone Python script.

- Applies hyperparameters from configs/exp/20260210_v1.yaml (training + phase1). => 401行目
- Uses base model Qwen/Qwen3-4B-Instruct-2507.
- Uploads LoRA adapter to HF_LORA_ADAP_REPO using HF_API from .env.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

os.environ.setdefault("XFORMERS_FORCE_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from transformers import TrainingArguments, Trainer, TrainerCallback
from unsloth import FastLanguageModel, is_bfloat16_supported

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Utils
# -----------------------------

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def load_config(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_openai_messages(ds: Dataset, msg_col: str = "messages") -> None:
    row0 = ds[0]
    ex = row0.get(msg_col, None)
    if not isinstance(ex, list):
        raise ValueError(f"Dataset must have list-style 'messages'. Got {type(ex)}")


def has_any_nonempty_assistant_turn(msgs: List[Dict[str, Any]]) -> bool:
    return any(
        m.get("role") == "assistant" and str(m.get("content", "")).strip() != ""
        for m in msgs
    )


def ends_with_nonempty_assistant(ex: Dict[str, Any]) -> bool:
    msgs = ex.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    c = msgs[-1].get("content", "")
    return isinstance(c, str) and c.strip() != ""


def shuffle_split(ds: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    ds_shuf = ds.shuffle(seed=seed)
    n = len(ds_shuf)
    n_val = max(1, int(round(n * val_ratio)))
    return ds_shuf.select(range(n_val, n)), ds_shuf.select(range(n_val))


def make_text_cache_builder(tokenizer):
    def _build(batch):
        full_out = []
        prefix_out = []
        full_len_out = []
        prefix_len_out = []

        for msgs in batch["messages"]:
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

            full_out.append(full)
            prefix_out.append(prefix)

            full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]

            full_len_out.append(len(full_ids))
            prefix_len_out.append(len(prefix_ids))

        return {
            "full_text": full_out,
            "prefix_text": prefix_out,
            "full_input_ids_len": full_len_out,
            "prefix_input_ids_len": prefix_len_out,
        }

    return _build


@dataclass
class AssistantOnlyCollatorCached:
    tokenizer: Any
    max_length: int

    def _find_subsequence(self, seq: List[int], sub: List[int]) -> int:
        if not sub or len(sub) > len(seq):
            return -1
        for i in range(0, len(seq) - len(sub) + 1):
            if seq[i : i + len(sub)] == sub:
                return i
        return -1

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        full_texts = [ex["full_text"] for ex in batch]
        prefix_texts = [ex["prefix_text"] for ex in batch]

        old_trunc = getattr(tok, "truncation_side", "right")
        old_pad = getattr(tok, "padding_side", "right")
        tok.truncation_side = "left"
        tok.padding_side = "right"

        try:
            full_enc_tr = tok(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids = full_enc_tr["input_ids"]
            attention_mask = full_enc_tr["attention_mask"]
            labels = torch.full_like(input_ids, fill_value=-100)

            full_ids_nt = tok(
                full_texts,
                return_tensors=None,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            prefix_ids_nt = tok(
                prefix_texts,
                return_tensors=None,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            marker_token_seqs = []
            if MASK_COT and OUTPUT_MARKERS:
                for m in OUTPUT_MARKERS:
                    mid = tok(m, add_special_tokens=False, truncation=False)["input_ids"]
                    if not mid:
                        continue
                    mid_nl = tok(m + "\n", add_special_tokens=False, truncation=False)["input_ids"]
                    mid_crlf = tok(m + "\r\n", add_special_tokens=False, truncation=False)["input_ids"]
                    marker_token_seqs.append((mid, mid_nl, mid_crlf))

            for i in range(input_ids.size(0)):
                trunc_left = max(0, len(full_ids_nt[i]) - self.max_length)
                boundary = len(prefix_ids_nt[i]) - trunc_left
                full_len_tr = int(attention_mask[i].sum().item())

                if boundary <= 0 or boundary >= full_len_tr:
                    continue

                span_start = boundary
                span_end = full_len_tr

                learn_start = span_start

                if MASK_COT and marker_token_seqs:
                    visible_ids = input_ids[i, :full_len_tr].tolist()
                    assistant_ids = visible_ids[span_start:span_end]

                    best_out = None  # (out_pos, after_pos)
                    for mid, mid_nl, mid_crlf in marker_token_seqs:
                        p = self._find_subsequence(assistant_ids, mid_nl)
                        if p != -1:
                            out_pos = span_start + p
                            after_pos = out_pos + len(mid_nl)
                        else:
                            p = self._find_subsequence(assistant_ids, mid_crlf)
                            if p != -1:
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid_crlf)
                            else:
                                p = self._find_subsequence(assistant_ids, mid)
                                if p == -1:
                                    continue
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid)

                        if (best_out is None) or (out_pos < best_out[0]):
                            best_out = (out_pos, after_pos)

                    if best_out is not None:
                        out_pos, after_pos = best_out
                        if OUTPUT_LEARN_MODE == "from_marker":
                            learn_start = out_pos
                        else:
                            learn_start = after_pos
                        learn_start = max(span_start, min(learn_start, span_end))

                if learn_start < span_end:
                    labels[i, learn_start:span_end] = input_ids[i, learn_start:span_end]

            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        finally:
            tok.truncation_side = old_trunc
            tok.padding_side = old_pad


@torch.no_grad()
def filter_has_supervision(ds, collator):
    keep = []
    for i in range(len(ds)):
        out = collator([ds[i]])
        if (out["labels"][0] != -100).sum().item() > 0:
            keep.append(i)
    return ds.select(keep)


def count_all_masked(ds, collator, n=200, seed=3407):
    rng = random.Random(seed)
    n = min(n, len(ds))
    idxs = [rng.randrange(0, len(ds)) for _ in range(n)]
    all_masked = 0
    for i in idxs:
        out = collator([ds[i]])
        labels = out["labels"][0]
        if (labels != -100).sum().item() == 0:
            all_masked += 1
    print(f"[CHECK] all-masked samples in {n}: {all_masked} ({all_masked/max(1,n):.1%})")


class LabelStatsCallback(TrainerCallback):
    def __init__(self, dataset, collator, name="train", every_n_steps=100):
        self.dataset, self.collator, self.name, self.every_n_steps = dataset, collator, name, every_n_steps

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % self.every_n_steps) == 0:
            batch = [self.dataset[random.randint(0, len(self.dataset) - 1)] for _ in range(8)]
            out = self.collator(batch)
            valid = (out["labels"] != -100).sum().item()
            total = (out["attention_mask"] == 1).sum().item()
            print(f"\n[LabelStats:{self.name}] step={state.global_step} valid_ratio={valid/max(1,total):.4f}")


# -----------------------------
# README generation
# -----------------------------

def fmt_lr(x: float) -> str:
    try:
        return f"{float(x):.0e}"
    except Exception:
        return str(x)


def write_readme(
    output_dir: Path,
    base_model_id: str,
    dataset_ids: List[str],
    max_seq_len: int,
    epochs: float,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    hf_repo: str,
) -> None:
    dataset_yaml = "\n".join(f"- {d}" for d in dataset_ids)
    lr_str = fmt_lr(lr)
    text = f"""---
base_model: {base_model_id}
datasets:
{dataset_yaml}
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
**{base_model_id}** using **QLoRA (4-bit, Unsloth)**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.

## Training Objective

This adapter is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.

## Training Configuration

- Base model: {base_model_id}
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

base = "{base_model_id}"
adapter = "{hf_repo}"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & Terms (IMPORTANT)

Training data: {', '.join(dataset_ids)}

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

MASK_COT = _getenv("SFT_MASK_COT", "1") in ("1", "true", "True")
OUTPUT_MARKERS = [
    s.strip()
    for s in _getenv(
        "SFT_OUTPUT_MARKERS",
        "Output:,OUTPUT:,Final:,Answer:,Result:,Response:",
    ).split(",")
    if s.strip()
]
OUTPUT_LEARN_MODE = _getenv("SFT_OUTPUT_LEARN_MODE", "after_marker")


def main() -> None:
    load_dotenv(ROOT / ".env")

    cfg = load_config(ROOT / "configs/exp/20260211_v1.yaml")

    base_model_id = cfg["model"]["base_model"]
    phase1 = cfg["phase1"]
    training = cfg["training"]
    lora = cfg["lora"]

    # Dataset path is fixed for this run (not a hyperparameter).
    dataset_path = ROOT / "data/processed/20260209_v1/train_daichira.jsonl"
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = ROOT / "outputs/models/20260210_v2/phase1"

    # Non-hyperparameter defaults (fixed to match existing pipeline behavior).
    seed = 42
    val_ratio = 0.1

    max_seq_len = int(training["max_length"])
    per_device_bs = int(training["batch_size"])
    grad_accum = int(training["grad_accum"])
    save_steps = int(training["save_steps"])
    eval_steps = int(training["eval_steps"])

    epochs = float(phase1["epochs"])
    lr = float(phase1["learning_rate"])

    # LoRA settings (from config).
    lora_r = int(lora["r"])
    lora_alpha = int(lora["alpha"])
    lora_target_modules = list(lora["target_modules"])

    precision = str(training.get("precision", "bf16")).lower()

    hf_token = os.environ.get("HF_API")
    hf_repo = os.environ.get("HF_LORA_ADAP_REPO")
    if not hf_token:
        raise RuntimeError("HF_API is missing in .env")
    if not hf_repo:
        raise RuntimeError("HF_LORA_ADAP_REPO is missing in .env")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from {dataset_path}")
    ds_all = load_dataset("json", data_files=str(dataset_path), split="train")

    ensure_openai_messages(ds_all)

    ds_all = ds_all.filter(lambda ex: has_any_nonempty_assistant_turn(ex["messages"]))
    ds_all = ds_all.filter(ends_with_nonempty_assistant)

    train_ds, val_ds = shuffle_split(ds_all, val_ratio, seed)

    print("[INFO] Loading base model:", base_model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    build_cache = make_text_cache_builder(tokenizer)
    train_ds = train_ds.map(build_cache, batched=True, num_proc=1, desc="Caching train")
    val_ds = val_ds.map(build_cache, batched=True, num_proc=1, desc="Caching val")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        use_gradient_checkpointing=training.get("gradient_checkpointing", "unsloth"),
        random_state=seed,
    )

    use_bf16 = precision == "bf16" and is_bfloat16_supported()
    use_fp16 = not use_bf16

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        max_steps=-1,
        bf16=use_bf16,
        fp16=use_fp16,
        push_to_hub=False,
        report_to="none",
        group_by_length=False,
        remove_unused_columns=False,
    )

    collator = AssistantOnlyCollatorCached(tokenizer=tokenizer, max_length=max_seq_len)

    print("[INFO] Checking all-masked samples before filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=seed)

    print("[INFO] Filtering train/val to remove all-masked samples...")
    train_ds = filter_has_supervision(train_ds, collator)
    val_ds = filter_has_supervision(val_ds, collator)

    print("[INFO] New sizes:", "train =", len(train_ds), "val =", len(val_ds))
    print("[INFO] Checking all-masked samples after filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=seed)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.add_callback(LabelStatsCallback(train_ds, collator, name="train", every_n_steps=10))

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving adapter & tokenizer...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[INFO] Done. Saved to {output_dir}")

    dataset_ids = [
        "daichira/structured-3k-mix-sft",
        "daichira/structured-5k-mix-sft",
        "daichira/structured-hard-sft-4k",
    ]
    write_readme(
        output_dir=Path(output_dir),
        base_model_id=base_model_id,
        dataset_ids=dataset_ids,
        max_seq_len=max_seq_len,
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        hf_repo=hf_repo,
    )

    print("[INFO] Logging into Hugging Face...")
    login(token=hf_token)
    api = HfApi()

    required_files = {
        "adapter_config.json",
        "README.md",
    }
    present = {p.name for p in Path(output_dir).iterdir() if p.is_file()}
    missing = [f for f in required_files if f not in present]
    if not any(f.startswith("adapter_model.") for f in present):
        missing.append("adapter_model.(safetensors|bin)")
    if missing:
        raise RuntimeError(
            "Upload aborted. Missing required files:\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    stage_dir = Path("/tmp/hf_upload_stage")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "README.md",
        "adapter_config.json",
        "adapter_model.*",
        "tokenizer.*",
        "special_tokens_map.json",
        "*.json",
    ]

    def is_allowed(name: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(name, pat) for pat in allow_patterns)

    for p in Path(output_dir).iterdir():
        if p.is_file() and is_allowed(p.name):
            (stage_dir / p.name).write_bytes(p.read_bytes())

    print("[INFO] Uploading to Hugging Face:", hf_repo)
    api.create_repo(
        repo_id=hf_repo,
        repo_type="model",
        exist_ok=True,
        private=True,
    )
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message="Upload LoRA adapter (README written by author)",
    )
    print(f"[INFO] Upload done: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()
