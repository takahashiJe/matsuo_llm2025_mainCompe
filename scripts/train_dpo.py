#!/usr/bin/env python
"""
Unsloth + TRL DPO training script.

Example:
  python scripts/train_dpo.py \
    --config configs/train_lora.yaml \
    --adapter-path outputs/models/20260206-204735 \
    --output-dir outputs/dpo_runs/20260206-204735_dpo \
    --dataset u-10bei/dpo-dataset-qwen-cot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import set_seed
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training with Unsloth + TRL.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="u-10bei/dpo-dataset-qwen-cot")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--precision", type=str, default="")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, object]:
    if not path:
        return {}
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def apply_config_defaults(args: argparse.Namespace, cfg: Dict[str, object]) -> None:
    if not cfg:
        return
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    dpo_cfg = cfg.get("dpo", {})

    if getattr(args, "base_model", None) == "Qwen/Qwen3-4B-Instruct-2507":
        args.base_model = model_cfg.get("base_model", args.base_model)
    if getattr(args, "adapter_path", None) == "":
        args.adapter_path = model_cfg.get("adapter_path", args.adapter_path)
    if getattr(args, "dataset", None) == "u-10bei/dpo-dataset-qwen-cot":
        args.dataset = data_cfg.get("dataset", args.dataset)
    if getattr(args, "split", None) == "train":
        args.split = data_cfg.get("split", args.split)
    if getattr(args, "output_dir", None) == "":
        args.output_dir = train_cfg.get("output_dir", args.output_dir)
    if getattr(args, "max_length", None) == 1024:
        args.max_length = int(train_cfg.get("max_length", args.max_length))
    if getattr(args, "max_seq_length", None) == 2048:
        args.max_seq_length = int(train_cfg.get("max_length", args.max_seq_length))
    if getattr(args, "max_prompt_length", None) == 512:
        args.max_prompt_length = int(
            train_cfg.get("max_prompt_length", dpo_cfg.get("max_prompt_length", args.max_prompt_length))
        )
    if getattr(args, "epochs", None) == 1:
        args.epochs = int(train_cfg.get("epochs", args.epochs))
    if getattr(args, "batch_size", None) == 2:
        args.batch_size = int(train_cfg.get("batch_size", args.batch_size))
    if getattr(args, "grad_accum", None) == 4:
        args.grad_accum = int(train_cfg.get("grad_accum", args.grad_accum))
    if getattr(args, "learning_rate", None) == 1e-7:
        args.learning_rate = float(train_cfg.get("learning_rate", args.learning_rate))
    if getattr(args, "beta", None) == 0.1:
        args.beta = float(dpo_cfg.get("beta", args.beta))
    if getattr(args, "logging_steps", None) == 50:
        args.logging_steps = int(train_cfg.get("logging_steps", args.logging_steps))
    precision = str(train_cfg.get("precision", "")).lower()
    if precision in {"bf16", "fp16", "none", "auto"}:
        args.precision = precision
    args._lora_cfg = lora_cfg
    if not args.gradient_checkpointing:
        args.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    if getattr(args, "max_prompt_length", None) == 512 and args.max_length != 1024:
        args.max_prompt_length = max(1, args.max_length // 2)


def format_dataset(tokenizer, dataset):
    def _fmt(examples) -> Dict[str, List[str]]:
        new_prompts, new_chosens, new_rejecteds = [], [], []
        for prompt, chosen, rejected in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_chosen = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": chosen}],
                tokenize=False,
            )
            formatted_rejected = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": rejected}],
                tokenize=False,
            )
            new_prompts.append(formatted_prompt)
            new_chosens.append(formatted_chosen)
            new_rejecteds.append(formatted_rejected)
        return {"prompt": new_prompts, "chosen": new_chosens, "rejected": new_rejecteds}

    keep_cols = {"prompt", "chosen", "rejected"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    return dataset.map(_fmt, batched=True, remove_columns=drop_cols)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_config_defaults(args, cfg)
    set_seed(args.seed)

    if not args.output_dir:
        raise SystemExit("output_dir is required (set --output-dir or training.output_dir in config).")
    if not args.adapter_path:
        raise SystemExit("adapter_path is required (set --adapter-path or model.adapter_path in config).")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = format_dataset(tokenizer, dataset)

    bf16_supported = is_bfloat16_supported()
    fp16 = not bf16_supported
    bf16 = bf16_supported
    if getattr(args, "precision", "") == "fp16":
        fp16, bf16 = True, False
    elif getattr(args, "precision", "") == "bf16":
        if bf16_supported:
            fp16, bf16 = False, True
        else:
            fp16, bf16 = False, False
    elif getattr(args, "precision", "") == "none":
        fp16, bf16 = False, False

    dpo_config = DPOConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        report_to="none",
    )

    # TRL versions differ: some accept `tokenizer`, others use `processing_class`.
    # Build kwargs dynamically to avoid version mismatch errors.
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "train_dataset": dataset,
        "args": dpo_config,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    import inspect

    sig = inspect.signature(DPOTrainer.__init__)
    allowed = set(sig.parameters.keys())
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in allowed}
    trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
