#!/usr/bin/env python
"""
Phase1-2 training runner for experiment 20260209_v1.

Phase1: SFT on daichira (clean, high-quality)
Phase2: SFT on u-10bei with rehearsal mix of daichira (sampled)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_20260209_v1.py \
  --config configs/exp/20260211_v2.yaml \
  --start-phase 1 \
  --single-run-phase1-2

"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

os.environ.setdefault("XFORMERS_FORCE_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch

# Force PyTorch to use math SDPA and avoid xformers/flash kernels.
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from train_lora import run_training  # noqa: E402

from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402


@dataclass
class PhaseConfig:
    epochs: float
    learning_rate: float


@dataclass
class DPOConfigLite:
    dataset_path: str
    max_seq_length: int
    max_length: int
    max_prompt_length: int
    epochs: int
    learning_rate: float
    beta: float
    batch_size: int
    grad_accum: int
    logging_steps: int
    precision: str
    gradient_checkpointing: bool
    max_samples: int


def load_config(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def read_jsonl_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def write_jsonl_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


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


def getenv_required(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"環境変数 {key} が未設定です。")
    return value


def render_readme(
    template_path: Path,
    dataset_id: str,
    max_seq_len: int,
    epochs: float,
    lr_str: str,
    lora_r: int,
    lora_alpha: int,
    hf_lora_repo: str | None = None,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    rendered = (
        text.replace("{dataset_id}", dataset_id)
        .replace("{max_seq_len}", str(max_seq_len))
        .replace("{epochs}", str(epochs))
        .replace("{lr_str}", lr_str)
        .replace("{lora_r}", str(lora_r))
        .replace("{lora_alpha}", str(lora_alpha))
    )
    if hf_lora_repo:
        rendered = rendered.replace("{hf_lora_repo}", hf_lora_repo)
    if not rendered.lstrip().startswith("---"):
        rendered = "---\n" + rendered
    return rendered


def sample_lines(lines: List[str], ratio: float, seed: int) -> List[str]:
    if ratio <= 0:
        return []
    if ratio >= 1:
        return list(lines)
    rng = random.Random(seed)
    k = max(1, int(round(len(lines) * ratio)))
    indices = rng.sample(range(len(lines)), k)
    return [lines[i] for i in indices]


def build_phase2_dataset(
    u10bei_path: Path,
    daichira_path: Path,
    daichira_ratio: float,
    seed: int,
    output_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    u10bei_lines = read_jsonl_lines(u10bei_path)
    daichira_lines = read_jsonl_lines(daichira_path)
    daichira_sample = sample_lines(daichira_lines, daichira_ratio, seed)

    mixed = list(u10bei_lines) + list(daichira_sample)
    rng = random.Random(seed)
    rng.shuffle(mixed)

    write_jsonl_lines(output_path, mixed)
    return output_path


def make_phase_config(
    base_cfg: Dict[str, object],
    phase_cfg: PhaseConfig,
    data_path: str,
    output_dir: str,
) -> Dict[str, object]:
    cfg = {
        "model": dict(base_cfg["model"]),
        "data": {
            "data_path": data_path,
            "second_data_path": "",
            "eval_ratio": base_cfg["data"]["eval_ratio"],
            "seed": base_cfg["data"]["seed"],
        },
        "training": dict(base_cfg["training"]),
        "lora": dict(base_cfg["lora"]),
    }
    cfg["training"]["output_dir"] = output_dir
    cfg["training"]["epochs"] = phase_cfg.epochs
    cfg["training"]["learning_rate"] = phase_cfg.learning_rate
    return cfg


def merge_lora_adapter(
    lora_dir: Path,
    output_dir: Path,
    base_model: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_shard_size: str = "5GB",
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map[dtype],
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    peft_model = PeftModel.from_pretrained(model, str(lora_dir))
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    readme = lora_dir / "README.md"
    if readme.exists():
        (output_dir / "README.md").write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")


def upload_merged_to_hf(
    model_dir: Path,
    repo_env: str = "HF_LORA_REPO",
    stage_dir: Path | None = None,
) -> List[str]:
    stage_arg = []
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_arg = ["--stage-dir", str(stage_dir)]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "upload_hf.py"),
        "--merged",
        "--model-dir",
        str(model_dir),
        "--repo-env",
        repo_env,
    ] + stage_arg
    subprocess.run(cmd, check=True)
    if stage_dir is None:
        return []
    return sorted(p.name for p in stage_dir.iterdir() if p.is_file())


def wait_for_hf_upload(repo_id: str, expected_files: List[str], timeout_sec: int = 1800) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_API"))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        if all(name in files for name in expected_files):
            return
        time.sleep(10)
    missing = [f for f in expected_files if f not in files]
    raise RuntimeError(f"HF upload not visible yet. Missing files: {missing}")


def merge_and_upload_phase2(
    *,
    base_output_dir: Path,
    phase2_out: Path,
    base_model: str,
    phase2_cfg: PhaseConfig,
    lora_cfg: Dict[str, object],
    max_length: int,
) -> None:
    print("=== Phase2 Merge: base + phase2 adapter ===")
    merged_dir = base_output_dir / "phase2_merged"
    merge_lora_adapter(
        lora_dir=phase2_out,
        output_dir=merged_dir,
        base_model=base_model,
        dtype="bfloat16",
    )
    readme = merged_dir / "README.md"
    if not readme.exists():
        template = ROOT / "README_Merged_Template.md"
        if not template.exists():
            raise RuntimeError(f"README template not found: {template}")
        dataset_id = "local/20260209_v1 (train_u10bei + train_daichira 10%)"
        readme_text = render_readme(
            template_path=template,
            dataset_id=dataset_id,
            max_seq_len=int(max_length),
            epochs=phase2_cfg.epochs,
            lr_str=str(phase2_cfg.learning_rate),
            lora_r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            hf_lora_repo=getenv_required("HF_LORA_REPO"),
        )
        readme.write_text(readme_text, encoding="utf-8")

    print("=== Upload merged model to HF_LORA_REPO ===")
    hf_lora_repo = getenv_required("HF_LORA_REPO")
    stage_dir = base_output_dir / "phase2_merged_upload_stage"
    expected_files = upload_merged_to_hf(
        merged_dir,
        repo_env="HF_LORA_REPO",
        stage_dir=stage_dir,
    )
    wait_for_hf_upload(
        repo_id=hf_lora_repo,
        expected_files=expected_files,
        timeout_sec=600,
    )


def should_use_chat_template(text: str) -> bool:
    markers = ("<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>")
    return not any(m in text for m in markers)


def format_dpo_samples(tokenizer, dataset) -> Tuple[object, int]:
    def _fmt(examples) -> Dict[str, List[str]]:
        new_prompts, new_chosens, new_rejecteds = [], [], []
        for prompt, chosen, rejected in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            if should_use_chat_template(prompt):
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = prompt

            if should_use_chat_template(chosen):
                formatted_chosen = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": chosen}],
                    tokenize=False,
                )
            else:
                formatted_chosen = chosen

            if should_use_chat_template(rejected):
                formatted_rejected = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": rejected}],
                    tokenize=False,
                )
            else:
                formatted_rejected = rejected

            new_prompts.append(formatted_prompt)
            new_chosens.append(formatted_chosen)
            new_rejecteds.append(formatted_rejected)
        return {"prompt": new_prompts, "chosen": new_chosens, "rejected": new_rejecteds}

    keep_cols = {"prompt", "chosen", "rejected"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    formatted = dataset.map(_fmt, batched=True, remove_columns=drop_cols)
    return formatted, len(formatted)


def run_phase3_dpo(
    base_model: str,
    lora_cfg: Dict[str, object],
    dpo_cfg: DPOConfigLite,
    output_dir: Path,
    seed: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"phase3 output already exists: {output_dir}")

    set_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=dpo_cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth" if dpo_cfg.gradient_checkpointing else False,
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    data_path = Path(dpo_cfg.dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"dpo dataset not found: {data_path}")
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if dpo_cfg.max_samples and dpo_cfg.max_samples > 0:
        dataset = dataset.select(range(min(dpo_cfg.max_samples, len(dataset))))
    dataset, _ = format_dpo_samples(tokenizer, dataset)

    bf16_supported = is_bfloat16_supported()
    fp16 = not bf16_supported
    bf16 = bf16_supported
    if dpo_cfg.precision == "fp16":
        fp16, bf16 = True, False
    elif dpo_cfg.precision == "bf16":
        if bf16_supported:
            fp16, bf16 = False, True
        else:
            fp16, bf16 = False, False
    elif dpo_cfg.precision == "none":
        fp16, bf16 = False, False

    dpo_config = DPOConfig(
        learning_rate=dpo_cfg.learning_rate,
        per_device_train_batch_size=dpo_cfg.batch_size,
        gradient_accumulation_steps=dpo_cfg.grad_accum,
        num_train_epochs=dpo_cfg.epochs,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16,
        bf16=bf16,
        logging_steps=dpo_cfg.logging_steps,
        output_dir=str(output_dir),
        beta=dpo_cfg.beta,
        max_length=dpo_cfg.max_length,
        max_prompt_length=dpo_cfg.max_prompt_length,
        seed=seed,
        report_to="none",
    )

    import inspect

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "train_dataset": dataset,
        "args": dpo_config,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    sig = inspect.signature(DPOTrainer.__init__)
    allowed = set(sig.parameters.keys())
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in allowed}
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase1-3 training for 20260209_v1.")
    parser.add_argument("--config", type=str, default="configs/exp/20260209_v1.yaml")
    parser.add_argument("--overwrite-phase2-data", action="store_true")
    parser.add_argument(
        "--single-run-phase1-2",
        action="store_true",
        help="Run phase1 then phase2 in a single training run (no adapter reload).",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a small sanity check for phases 1-3 (reduce dataset sizes only).",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Start from the given phase (1, 2, or 3).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    load_dotenv(Path(".env"))

    base_output_dir = Path(cfg["training"]["output_dir"])
    if args.sanity:
        base_output_dir = Path("outputs/models/20260209_v1_sanity")
        cfg["training"]["output_dir"] = str(base_output_dir)
    phase1_out = base_output_dir / "phase1"
    phase2_out = base_output_dir / "phase2"
    phase3_out = base_output_dir / "phase3"

    start_phase = int(args.start_phase)
    if start_phase <= 1 and not args.single_run_phase1_2:
        if phase1_out.exists() and any(phase1_out.iterdir()):
            raise RuntimeError(f"phase1 output already exists: {phase1_out}")
    elif start_phase == 2 and not args.single_run_phase1_2:
        if not phase1_out.exists() or not any(phase1_out.iterdir()):
            raise RuntimeError(f"phase1 output is required but missing: {phase1_out}")
    if start_phase <= 2 and phase2_out.exists() and any(phase2_out.iterdir()):
        raise RuntimeError(f"phase2 output already exists: {phase2_out}")
    if start_phase <= 3 and phase3_out.exists() and any(phase3_out.iterdir()):
        raise RuntimeError(f"phase3 output already exists: {phase3_out}")

    data_cfg = cfg["data"]
    seed = int(data_cfg["seed"])
    if start_phase <= 2:
        phase1_data = Path(data_cfg["phase1_data_path"])
        phase2_u10bei = Path(data_cfg["phase2_u10bei_path"])
        phase2_daichira = Path(data_cfg["phase2_daichira_path"])
        phase2_ratio = float(data_cfg["phase2_daichira_ratio"])

        if not phase1_data.exists():
            raise FileNotFoundError(f"phase1 data not found: {phase1_data}")
        if not phase2_u10bei.exists():
            raise FileNotFoundError(f"phase2 u10bei data not found: {phase2_u10bei}")
        if not phase2_daichira.exists():
            raise FileNotFoundError(f"phase2 daichira data not found: {phase2_daichira}")

        phase2_data_path = base_output_dir / "phase2_mixed.jsonl"
        phase2_data_path = build_phase2_dataset(
            u10bei_path=phase2_u10bei,
            daichira_path=phase2_daichira,
            daichira_ratio=phase2_ratio,
            seed=seed,
            output_path=phase2_data_path,
            overwrite=args.overwrite_phase2_data,
        )

    phase1_cfg = PhaseConfig(
        epochs=float(cfg["phase1"]["epochs"]),
        learning_rate=float(cfg["phase1"]["learning_rate"]),
    )
    phase2_cfg = PhaseConfig(
        epochs=float(cfg["phase2"]["epochs"]),
        learning_rate=float(cfg["phase2"]["learning_rate"]),
    )
    dpo_cfg = DPOConfigLite(
        dataset_path=str(cfg["dpo"]["dataset_path"]),
        max_seq_length=int(cfg["dpo"]["max_seq_length"]),
        max_length=int(cfg["dpo"]["max_length"]),
        max_prompt_length=int(cfg["dpo"]["max_prompt_length"]),
        epochs=int(cfg["dpo"]["epochs"]),
        learning_rate=float(cfg["dpo"]["learning_rate"]),
        beta=float(cfg["dpo"]["beta"]),
        batch_size=int(cfg["dpo"]["batch_size"]),
        grad_accum=int(cfg["dpo"]["grad_accum"]),
        logging_steps=int(cfg["dpo"]["logging_steps"]),
        precision=str(cfg["dpo"]["precision"]).lower(),
        gradient_checkpointing=bool(cfg["dpo"]["gradient_checkpointing"]),
        max_samples=int(cfg["dpo"]["max_samples"]),
    )
    if args.sanity and dpo_cfg.max_samples <= 0:
        dpo_cfg.max_samples = 128

    phase1_cfg_path = base_output_dir / "phase1_config.yaml"
    phase2_cfg_path = base_output_dir / "phase2_config.yaml"

    if start_phase <= 2:
        write_yaml(
            phase1_cfg_path,
            make_phase_config(
                base_cfg=cfg,
                phase_cfg=phase1_cfg,
                data_path=str(phase1_data),
                output_dir=str(base_output_dir),
            ),
        )
        write_yaml(
            phase2_cfg_path,
            make_phase_config(
                base_cfg=cfg,
                phase_cfg=phase2_cfg,
                data_path=str(phase2_data_path),
                output_dir=str(base_output_dir),
            ),
        )

    if start_phase <= 2 and args.single_run_phase1_2:
        print("=== Phase1+2: Single run (daichira -> u-10bei + rehearsal) ===")
        # Build a config that trains phase1 then phase2 sequentially without reloading adapters.
        combined_cfg = make_phase_config(
            base_cfg=cfg,
            phase_cfg=phase1_cfg,
            data_path=str(phase1_data),
            output_dir=str(base_output_dir),
        )
        combined_cfg["data"]["second_data_path"] = str(phase2_data_path)
        combined_cfg_path = base_output_dir / "phase1_2_config.yaml"
        write_yaml(combined_cfg_path, combined_cfg)
        run_training(
            config_path=str(combined_cfg_path),
            run_id="phase2",
            sanity=args.sanity,
            stage_output_names=["phase1", "phase2"],
            output_dir_override=str(base_output_dir),
        )
        merge_and_upload_phase2(
            base_output_dir=base_output_dir,
            phase2_out=phase2_out,
            base_model=cfg["model"]["base_model"],
            phase2_cfg=phase2_cfg,
            lora_cfg=cfg["lora"],
            max_length=int(cfg["training"]["max_length"]),
        )
    elif start_phase <= 1:
        print("=== Phase1: SFT (daichira) ===")
        run_training(
            config_path=str(phase1_cfg_path),
            run_id="phase1",
            sanity=args.sanity,
        )

    if start_phase <= 2 and not args.single_run_phase1_2:
        print("=== Phase2: SFT (u-10bei + daichira rehearsal) ===")
        run_training(
            config_path=str(phase2_cfg_path),
            run_id="phase2",
            adapter_path=str(phase1_out),
            sanity=args.sanity,
        )

        merge_and_upload_phase2(
            base_output_dir=base_output_dir,
            phase2_out=phase2_out,
            base_model=cfg["model"]["base_model"],
            phase2_cfg=phase2_cfg,
            lora_cfg=cfg["lora"],
            max_length=int(cfg["training"]["max_length"]),
        )

    if start_phase <= 3:
        print("=== Phase3: DPO (base = HF_LORA_REPO) ===")
        hf_lora_repo = getenv_required("HF_LORA_REPO")
        run_phase3_dpo(
            base_model=hf_lora_repo,
            lora_cfg=cfg["lora"],
            dpo_cfg=dpo_cfg,
            output_dir=phase3_out,
            seed=seed,
        )


if __name__ == "__main__":
    main()
