#!/usr/bin/env python
"""
TRL + QLoRA 学習スクリプト

ポイント:
- merged_train.jsonl を読み込み、train/eval を固定比率で分割
- assistant 出力内のタグ直後からのみ loss を計算
- QLoRA 設定で SFTTrainer を実行
"""

# python scripts/train_lora.py --config configs/exp/20260206_mix_v1.yaml --run-id 20260206_mix_v1

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import MethodType

if os.environ.get("UNSLOTH_FORCE_SDPA", "0") == "1":
    os.environ.setdefault("XFORMERS_FORCE_DISABLE", "1")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
else:
    # Ensure xFormers is not accidentally disabled via environment.
    os.environ.pop("XFORMERS_FORCE_DISABLE", None)
    os.environ.pop("XFORMERS_DISABLED", None)

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from unsloth import FastLanguageModel
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer


MASK_TAGS = ["Output:", "OUTPUT:", "Final:", "Answer:", "Result:", "Response:"]

if os.environ.get("UNSLOTH_FORCE_SDPA", "0") == "1":
    # Force PyTorch to use math SDPA and avoid xformers/flash kernels.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
else:
    # Explicitly allow xFormers when available.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass
    # Unsloth picks xformers if it's installed. Force SDPA to avoid xformers.
    try:
        import unsloth.utils.attention_dispatch as _attn_dispatch

        _attn_dispatch.HAS_XFORMERS = False
        _attn_dispatch.XFORMERS_BLOCK_DIAG_CLS = None
    except Exception:
        pass


def load_config(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def build_run_dir(base_dir: str, run_name_prefix: str = "") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path(base_dir) / f"{run_name_prefix}{ts}")


def resolve_output_dir(base_dir: str, run_id: Optional[str], run_name_prefix: str = "") -> str:
    if run_id:
        return str(Path(base_dir) / run_id)
    return build_run_dir(base_dir, run_name_prefix=run_name_prefix)


def messages_to_text(messages: List[dict]) -> str:
    """
    ChatML 風の単純なシリアライズ。
    役割の明示 + 内容で結合する。
    """
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def build_prompt_from_messages(messages: List[dict]) -> str:
    """
    生成用プロンプトを作る。
    - assistant 出力は除外し、最後に <|assistant|> を付与する
    """
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            break
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def extract_assistant_ref(messages: List[dict]) -> str:
    """参照回答（最初の assistant content）を取り出す。"""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""

def find_first_tag_pos(text: str, tags: List[str]) -> Optional[Tuple[int, str]]:
    """最初に出現したタグ位置（開始インデックス, タグ文字列）を返す。"""
    found = []
    for tag in tags:
        idx = text.find(tag)
        if idx >= 0:
            found.append((idx, tag))
    if not found:
        return None
    found.sort(key=lambda x: x[0])
    return found[0]


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    idx = int(round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def compute_length_stats(texts: List[str], tokenizer, max_length: int) -> Dict[str, int]:
    lengths: List[int] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        lengths.append(len(ids))
    lengths.sort()
    return {
        "samples": len(lengths),
        "p50": percentile(lengths, 50),
        "p75": percentile(lengths, 75),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p97": percentile(lengths, 97),
        "p98": percentile(lengths, 98),
        "p99": percentile(lengths, 99),
        "p99_5": percentile(lengths, 99.5),
        "max": lengths[-1] if lengths else 0,
    }


def save_yaml(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


@dataclass
class CollatorConfig:
    tokenizer: object
    max_length: int
    tags: List[str]


class OutputTagCollator:
    """
    Outputタグ以降のみ loss を計算するための data collator。
    - input_ids は全文
    - labels は Output タグ直後から input_ids をコピーし、それ以前は -100
    - タグが無い場合は全文を loss 対象
    - loss_mask_mode が "full" の場合は全文を loss 対象
    - loss_mask_mode が "output_only" の場合は Output タグ以降のみ
    - mode 未指定の場合は system ロールが含まれると全文対象（u-10bei系を想定）
    """

    def __init__(self, cfg: CollatorConfig) -> None:
        self.cfg = cfg

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        modes = [f.pop("loss_mask_mode", "auto") for f in features]
        tok = self.cfg.tokenizer.pad(
            features,
            padding=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        labels = input_ids.clone()

        # padding は常に loss 対象外
        labels[attention_mask == 0] = -100

        # 各サンプルでタグ直後までを -100 にする
        for i in range(input_ids.size(0)):
            valid_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :valid_len].tolist()
            text = self.cfg.tokenizer.decode(ids, skip_special_tokens=False)

            mode = modes[i] if i < len(modes) else "auto"
            if mode == "full":
                continue
            if mode == "auto" and "<|system|>" in text:
                continue

            hit = find_first_tag_pos(text, self.cfg.tags)
            if hit is None:
                # タグが無い場合は全文を学習対象
                continue

            tag_pos, tag = hit
            end = tag_pos + len(tag)
            # タグ直後の空白・改行も除外して学習対象を厳密に「タグ以降」にする
            while end < len(text) and text[end] in (" ", "\t", "\n", "\r"):
                end += 1
            prefix = text[:end]
            prefix_ids = self.cfg.tokenizer(
                prefix, truncation=True, max_length=self.cfg.max_length, add_special_tokens=False
            )["input_ids"]
            cut = min(len(prefix_ids), valid_len)
            labels[i, :cut] = -100

        tok["labels"] = labels
        return tok


class EvalSampleCallback(TrainerCallback):
    """
    各エポック終了時に固定サンプルの生成結果を保存する。
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        output_dir: str,
        sample_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        rng = torch.Generator().manual_seed(seed)
        n = len(dataset)
        if n == 0:
            self.indices = []
        else:
            perm = torch.randperm(n, generator=rng).tolist()
            self.indices = perm[: min(sample_size, n)]

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model_was_training = model.training
        model.eval()

        results = []
        device = next(model.parameters()).device
        for idx in self.indices:
            ex = self.dataset[idx]
            messages = ex.get("messages", [])
            prompt = build_prompt_from_messages(messages)
            ref = extract_assistant_ref(messages)

            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=False)
            results.append(
                {
                    "epoch": float(state.epoch),
                    "index": idx,
                    "prompt": prompt,
                    "prediction": text,
                    "reference": ref,
                }
            )

        out_path = self.output_dir / f"eval_samples_epoch{int(state.epoch)}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Release unused cached GPU memory after generation to reduce fragmentation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model_was_training:
            model.train()


def run_training(
    config_path: str = "configs/train_lora.yaml",
    data_path_override: Optional[str] = None,
    second_data_path_override: Optional[str] = None,
    third_data_path_override: Optional[str] = None,
    run_id: Optional[str] = None,
    adapter_path: Optional[str] = None,
    sanity: bool = False,
    sanity_train_size: int = 64,
    sanity_eval_size: int = 16,
    sanity_run_name_prefix: str = "sanity-",
    sanity_num_train_epochs: Optional[float] = None,
    sanity_logging_steps: Optional[int] = None,
    stage_output_names: Optional[List[str]] = None,
    output_dir_override: Optional[str] = None,
) -> None:
    cfg = load_config(Path(config_path))
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    model_cfg = cfg["model"]

    data_path = data_path_override if data_path_override else data_cfg["data_path"]
    second_data_path_cfg = data_cfg.get("second_data_path")
    third_data_path_cfg = data_cfg.get("third_data_path")
    second_data_path = (
        second_data_path_override
        if second_data_path_override is not None
        else (str(second_data_path_cfg) if second_data_path_cfg else None)
    )
    third_data_path = (
        third_data_path_override
        if third_data_path_override is not None
        else (str(third_data_path_cfg) if third_data_path_cfg else None)
    )
    eval_ratio = float(data_cfg["eval_ratio"])
    seed = int(data_cfg["seed"])

    base_model = model_cfg["base_model"]
    run_name_prefix = sanity_run_name_prefix if sanity else ""
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else resolve_output_dir(train_cfg["output_dir"], run_id, run_name_prefix=run_name_prefix)
    )
    run_name = Path(output_dir).name
    max_length = int(train_cfg["max_length"])
    effective_epochs = (
        float(sanity_num_train_epochs)
        if sanity and sanity_num_train_epochs is not None
        else float(train_cfg["epochs"])
    )
    effective_logging_steps = (
        int(sanity_logging_steps)
        if sanity and sanity_logging_steps is not None
        else 50
    )

    stage_data_paths = [data_path]
    if second_data_path:
        stage_data_paths.append(second_data_path)
    if third_data_path:
        stage_data_paths.append(third_data_path)
    print(f"Stage datasets: {stage_data_paths}")

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not stage_output_names:
            raise RuntimeError(f"出力先が既に存在します。別の --run-id を指定してください: {output_dir}")

    # Unsloth でモデル・トークナイザを初期化
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Unsloth + QLoRA (gradient checkpointing は unsloth 版)
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"adapter_path not found: {adapter_dir}")
        # Ensure adapter is not saved as inference-only.
        adapter_cfg_path = adapter_dir / "adapter_config.json"
        if adapter_cfg_path.exists():
            try:
                adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
                if adapter_cfg.get("inference_mode") is True:
                    adapter_cfg["inference_mode"] = False
                    adapter_cfg_path.write_text(
                        json.dumps(adapter_cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        # Expose Unsloth training/inference toggles on the PEFT wrapper if present on base.
        try:
            base = None
            if hasattr(model, "get_base_model"):
                base = model.get_base_model()
            elif hasattr(model, "base_model"):
                base = model.base_model
            elif hasattr(model, "model"):
                base = model.model
            if base is not None and hasattr(base, "for_training"):
                model.for_training = base.for_training
                if hasattr(base, "for_inference"):
                    model.for_inference = base.for_inference
        except Exception:
            pass
        # Ensure adapter is active and trainable for phase2.
        adapter_name = None
        if hasattr(model, "peft_config"):
            if isinstance(model.peft_config, dict) and model.peft_config:
                adapter_name = next(iter(model.peft_config.keys()))
                # Override inference_mode if saved as inference-only.
                try:
                    model.peft_config[adapter_name].inference_mode = False
                except Exception:
                    pass
        try:
            if adapter_name and hasattr(model, "set_adapter"):
                model.set_adapter(adapter_name)
        except Exception:
            pass
        try:
            model.enable_adapter_layers()
        except Exception:
            pass
        try:
            if hasattr(model, "for_training"):
                model.for_training()
        except Exception:
            pass
        model.train()
        # Safety: ensure LoRA params are trainable even if PEFT defaults differ.
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        # Fail fast if nothing is trainable.
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters after loading adapter.")
        print(f"[debug] Trainable params (count): {len(trainable)}")
        print(f"[debug] Trainable params (sample): {trainable[:5]}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            target_modules=list(lora_cfg["target_modules"]),
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
        )

    stage_records = []
    single_stage = len(stage_data_paths) == 1
    for stage_idx, stage_data_path in enumerate(stage_data_paths, start=1):
        if single_stage:
            stage_name = "main"
            stage_run_name = run_name
            stage_output_dir = Path(output_dir)
        else:
            if stage_output_names and stage_idx - 1 < len(stage_output_names):
                stage_name = stage_output_names[stage_idx - 1]
            else:
                stage_name = f"stage{stage_idx}"
            # Use clean run names for W&B when custom stage names are provided.
            stage_run_name = stage_name if stage_output_names else f"{run_name}-{stage_name}"
            stage_output_dir = Path(output_dir) / stage_name
        print(f"[{stage_name}] Loading dataset: {stage_data_path}")

        ds_raw = load_dataset("json", data_files=stage_data_path, split="train")
        ds_raw = ds_raw.train_test_split(test_size=eval_ratio, seed=seed)
        if sanity:
            raw_train = ds_raw["train"]
            raw_eval = ds_raw["test"]
            ds_raw["train"] = raw_train.select(range(min(sanity_train_size, len(raw_train))))
            ds_raw["test"] = raw_eval.select(range(min(sanity_eval_size, len(raw_eval))))
            print(
                f"[{stage_name}][sanity] train={len(ds_raw['train'])} eval={len(ds_raw['test'])} "
                f"(limits: train={sanity_train_size}, eval={sanity_eval_size})"
            )

        def format_fn(example: Dict[str, object]) -> Dict[str, str]:
            messages = example.get("messages", [])
            meta = example.get("metadata") or {}
            mode = str(meta.get("loss_mask_mode", "auto"))
            return {"text": messages_to_text(messages), "loss_mask_mode": mode}

        ds = ds_raw.map(format_fn, remove_columns=ds_raw["train"].column_names)

        # タグ出現率の簡易ログ
        def has_tag(text: str) -> bool:
            return any(tag in text for tag in MASK_TAGS)

        tag_hits = sum(1 for t in ds["train"]["text"] if has_tag(t))
        print(f"[{stage_name}] Tag hit ratio (train): {tag_hits}/{len(ds['train'])}")

        run_meta = {
            "is_sanity": sanity,
            "base_model": base_model,
            "data_path": stage_data_path,
            "stage_index": stage_idx,
            "num_stages": len(stage_data_paths),
            "eval_ratio": eval_ratio,
            "seed": seed,
            "max_length": max_length,
            "batch_size": int(train_cfg["batch_size"]),
            "grad_accum": int(train_cfg["grad_accum"]),
            "learning_rate": float(train_cfg["learning_rate"]),
            "epochs": effective_epochs,
            "logging_steps": effective_logging_steps,
            "precision": str(train_cfg["precision"]),
            "packing": bool(train_cfg["packing"]),
            "lora_r": int(lora_cfg["r"]),
            "lora_alpha": int(lora_cfg["alpha"]),
            "lora_target_modules": list(lora_cfg["target_modules"]),
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bf16",
            },
            "mask_tags": MASK_TAGS,
            "tag_hit_train": tag_hits,
            "train_size": len(ds["train"]),
            "eval_size": len(ds["test"]),
        }
        length_stats = compute_length_stats(ds["train"]["text"], tokenizer, max_length=max_length)
        save_yaml(stage_output_dir / "run_meta.yaml", run_meta)
        save_yaml(stage_output_dir / "length_stats.yaml", length_stats)
        stage_records.append(
            {"stage": stage_name, "data_path": stage_data_path, "output_dir": str(stage_output_dir)}
        )

        args = SFTConfig(
            output_dir=str(stage_output_dir),
            num_train_epochs=effective_epochs,
            per_device_train_batch_size=int(train_cfg["batch_size"]),
            per_device_eval_batch_size=int(train_cfg["batch_size"]),
            gradient_accumulation_steps=int(train_cfg["grad_accum"]),
            learning_rate=float(train_cfg["learning_rate"]),
            save_steps=int(train_cfg["save_steps"]),
            eval_steps=int(train_cfg["eval_steps"]),
            eval_strategy="steps",
            logging_steps=effective_logging_steps,
            bf16=str(train_cfg["precision"]).lower() == "bf16",
            fp16=str(train_cfg["precision"]).lower() == "fp16",
            report_to="wandb",
            run_name=stage_run_name,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_num_proc=2,
            do_train=True,
        )

        collator = OutputTagCollator(
            CollatorConfig(tokenizer=tokenizer, max_length=max_length, tags=MASK_TAGS)
        )
        # Debug: ensure labels are not all -100 (no loss) for a small batch.
        try:
            sample_batch = [ds["train"][i] for i in range(min(2, len(ds["train"])))]
            debug_inputs = [tokenizer(sample["text"], truncation=True, max_length=max_length) for sample in sample_batch]
            debug = collator(debug_inputs)
            valid_labels = (debug["labels"] != -100).sum().item()
            print(f"[debug] Valid label tokens in sample batch: {valid_labels}")
            if valid_labels == 0:
                raise RuntimeError(
                    "All labels are -100 in sample batch. Check Output tags or loss_mask_mode."
                )
        except Exception as e:
            raise RuntimeError(f"Label debug check failed: {e}")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            formatting_func=lambda x: x["text"],
            data_collator=collator,
            args=args,
            packing=bool(train_cfg["packing"]),
            max_seq_length=max_length,
            dataset_text_field="text",
            dataset_num_proc=2,
        )
        # Diagnostics: capture last inputs and intercept backward if loss has no grad.
        orig_compute_loss = trainer.compute_loss
        orig_backward = trainer.accelerator.backward

        def _compute_loss(self, model, inputs, *args, **kwargs):
            self._last_inputs = inputs
            return orig_compute_loss(model, inputs, *args, **kwargs)

        def _backward(loss, **kwargs):
            if isinstance(loss, torch.Tensor) and not loss.requires_grad:
                try:
                    model_ref = trainer.model
                    trainable = [n for n, p in model_ref.named_parameters() if p.requires_grad]
                    print("[debug][no_grad] torch.is_grad_enabled:", torch.is_grad_enabled())
                    print("[debug][no_grad] model.training:", getattr(model_ref, "training", None))
                    print("[debug][no_grad] trainable param count:", len(trainable))
                    inputs = getattr(trainer, "_last_inputs", None)
                    if inputs is not None:
                        for key in ("input_ids", "labels", "attention_mask"):
                            if key in inputs:
                                t = inputs[key]
                                print(f"[debug][no_grad] {key}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
                        dump = {
                            "input_ids": inputs.get("input_ids"),
                            "labels": inputs.get("labels"),
                            "attention_mask": inputs.get("attention_mask"),
                        }
                        dump_path = stage_output_dir / "no_grad_batch.pt"
                        torch.save(dump, dump_path)
                        print(f"[debug][no_grad] Saved batch dump to: {dump_path}")
                except Exception as e:
                    print(f"[debug][no_grad] Failed to dump diagnostics: {e}")
                raise RuntimeError("Loss has no grad (diagnostics dumped).")
            return orig_backward(loss, **kwargs)

        trainer.compute_loss = MethodType(_compute_loss, trainer)
        trainer.accelerator.backward = _backward
        # Debug: verify loss requires grad on a tiny batch.
        try:
            model.train()
            sample = [ds["train"][0]]
            sample_inputs = [tokenizer(sample[0]["text"], truncation=True, max_length=max_length)]
            batch = collator(sample_inputs)
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else None
            print(f"[debug] loss requires_grad: {getattr(loss, 'requires_grad', None)}")
            print(f"[debug] loss grad_fn: {getattr(loss, 'grad_fn', None)}")
            if loss is None or not getattr(loss, "requires_grad", False):
                raise RuntimeError("Loss does not require grad on a debug batch.")
        except Exception as e:
            raise RuntimeError(f"Loss grad debug check failed: {e}")
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg["early_stopping_patience"]),
                early_stopping_threshold=float(train_cfg["early_stopping_threshold"]),
            )
        )
        trainer.add_callback(
            EvalSampleCallback(
                dataset=ds_raw["test"],
                tokenizer=tokenizer,
                output_dir=str(stage_output_dir),
                sample_size=int(train_cfg["eval_sample_size"]),
                max_new_tokens=int(train_cfg["eval_max_new_tokens"]),
                temperature=float(train_cfg["eval_temperature"]),
                top_p=float(train_cfg["eval_top_p"]),
                seed=seed,
            )
        )
        trainer.train()
        trainer.save_model(str(stage_output_dir))
        print(f"[{stage_name}] Saved stage model to: {stage_output_dir}")

    save_yaml(Path(output_dir) / "stage_plan.yaml", {"stages": stage_records})
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved final model to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL QLoRA trainer.")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data.data_path from config",
    )
    parser.add_argument(
        "--second-data-path",
        type=str,
        default=None,
        help="Optional second stage data path; trained after --data-path",
    )
    parser.add_argument(
        "--third-data-path",
        type=str,
        default=None,
        help="Optional third stage data path; trained after --second-data-path",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="outputs/models/{run_id} に固定して出力する",
    )
    args = parser.parse_args()
    run_training(
        config_path=args.config,
        data_path_override=args.data_path,
        second_data_path_override=args.second_data_path,
        third_data_path_override=args.third_data_path,
        run_id=args.run_id,
        sanity=False,
    )


if __name__ == "__main__":
    main()
