#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_lora_dir(run_id: str | None, lora_dir: str | None) -> Path:
    if lora_dir:
        return Path(lora_dir)
    if run_id:
        return Path("outputs") / "models" / run_id
    raise RuntimeError("--run-id か --lora-dir のどちらかを指定してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--run-id", help="outputs/models/{run_id} の run_id")
    parser.add_argument("--lora-dir", help="LoRA アダプタの保存先ディレクトリ")
    parser.add_argument(
        "--output-dir",
        help="マージ後の出力ディレクトリ (default: {lora_dir}_merged)",
    )
    parser.add_argument(
        "--base-model",
        help="ベースモデル名/パス (未指定なら adapter_config から取得)",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="bfloat16",
        help="読み込みdtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map for transformers (default: auto, use 'cpu' for CPU only)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="save_pretrained の max_shard_size (default: 5GB)",
    )
    args = parser.parse_args()

    lora_dir = _resolve_lora_dir(args.run_id, args.lora_dir)
    if not lora_dir.exists():
        raise RuntimeError(f"LoRA ディレクトリが見つかりません: {lora_dir}")

    peft_cfg = PeftConfig.from_pretrained(str(lora_dir))
    base_model = args.base_model or peft_cfg.base_model_name_or_path
    if not base_model:
        raise RuntimeError("ベースモデルが特定できません。--base-model を指定してください。")

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{lora_dir}_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = DTYPE_MAP[args.dtype]
    print(f"Loading base model: {base_model} (dtype={args.dtype}, device_map={args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    print(f"Loading LoRA adapter: {lora_dir}")
    peft_model = PeftModel.from_pretrained(model, str(lora_dir))

    print("Merging...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    readme = lora_dir / "README.md"
    if readme.exists():
        shutil.copy2(readme, output_dir / "README.md")

    print("✅ Merge complete.")


if __name__ == "__main__":
    main()
