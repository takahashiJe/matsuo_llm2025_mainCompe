#!/usr/bin/env python3

# python scripts/upload_hf.py --merged --model-dir outputs/models/20260206-005340_merged
# python scripts/upload_hf.py --run-id 20260206-005340

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _getenv(key: str, default: str | None = None) -> str:
    value = os.environ.get(key, default)
    if value is None or value == "":
        raise RuntimeError(f"環境変数 {key} が未設定です。")
    return value


def _resolve_model_dir(run_id: str | None, model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    if run_id:
        return Path("outputs") / "models" / run_id
    raise RuntimeError("--run-id か --model-dir のどちらかを指定してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to Hugging Face Hub")
    parser.add_argument("--run-id", help="outputs/models/{run_id} の run_id")
    parser.add_argument("--model-dir", help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="マージ済みモデルとしてアップロードする",
    )
    parser.add_argument(
        "--stage-dir",
        default="outputs/hf_upload_stage",
        help="アップロード用の一時ディレクトリ (default: outputs/hf_upload_stage)",
    )
    args = parser.parse_args()

    _load_dotenv(Path(".env"))

    lora_save_dir = _resolve_model_dir(args.run_id, args.model_dir)
    if not lora_save_dir.exists():
        raise RuntimeError(f"モデル保存ディレクトリが見つかりません: {lora_save_dir}")

    hf_token = _getenv("HF_API")
    hf_repo_id = _getenv("HF_REPO")
    private = _getenv("HF_PRIVATE", "1") in ("1", "true", "True")

    api = HfApi(token=hf_token)

    # 3.1) 必須ファイルの存在確認
    present = {p.name for p in lora_save_dir.iterdir() if p.is_file()}
    if args.merged:
        required_files = {"config.json", "README.md"}
        missing = [f for f in required_files if f not in present]
        if not (
            any(fnmatch.fnmatch(name, "model.*") for name in present)
            or any(fnmatch.fnmatch(name, "pytorch_model.*") for name in present)
        ):
            missing.append("model.(safetensors|bin) or pytorch_model.(bin|safetensors)")
    else:
        required_files = {"adapter_config.json", "README.md"}
        missing = [f for f in required_files if f not in present]
        if not any(name.startswith("adapter_model.") for name in present):
            missing.append("adapter_model.(safetensors|bin)")
    if missing:
        raise RuntimeError(
            "アップロードを中止しました。\n"
            "以下の必須ファイルが見つかりません:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nアップロード前に、README.md を手書きで作成し保存してください。"
        )

    print("✅ 必須ファイルの確認が完了しました。")

    # 3.2) アップロード対象の選別（ホワイトリスト）
    if args.merged:
        allow_patterns = [
            "README.md",
            "config.json",
            "model.*",
            "pytorch_model.*",
            "chat_template.jinja",
            "tokenizer.*",
            "special_tokens_map.json",
            "*.json",
        ]
    else:
        allow_patterns = [
            "README.md",
            "adapter_config.json",
            "adapter_model.*",
            "tokenizer.*",
            "special_tokens_map.json",
            "*.json",
        ]

    def is_allowed(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in allow_patterns)

    stage_dir = Path(args.stage_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    for p in lora_save_dir.iterdir():
        if p.is_file() and is_allowed(p.name):
            if args.merged and p.name == "tokenizer_config.json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if not data.get("chat_template"):
                    tmpl = lora_save_dir / "chat_template.jinja"
                    if tmpl.exists():
                        data["chat_template"] = tmpl.read_text(encoding="utf-8")
                (stage_dir / p.name).write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                (stage_dir / p.name).write_bytes(p.read_bytes())

    print("📦 アップロード対象ファイル:", [p.name for p in stage_dir.iterdir()])

    # 3.3) リポジトリ作成とアップロード
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
    )

    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload LoRA adapter (README written by author)",
    )

    print("✅ アップロードが正常に完了しました。")
    print(f"URL: https://huggingface.co/{hf_repo_id}")


if __name__ == "__main__":
    main()
