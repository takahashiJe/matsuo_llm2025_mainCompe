#!/usr/bin/env python
"""
ローカルの tokenizer を使ってデータセットのトークン長分布を計測する。

前提:
- tokenizer は `models/Qwen3-4B-Instruct-2507` に配置済み
- data_path は configs/train_lora.yaml の data.data_path を利用
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import yaml
from transformers import AutoTokenizer


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def messages_to_text(messages: List[dict]) -> str:
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    idx = int(round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def main() -> None:
    cfg = load_config(Path("configs/train_lora.yaml"))
    data_path = Path(cfg["data"]["data_path"])

    tok_path = Path("models/Qwen3-4B-Instruct-2507")
    if not tok_path.exists():
        raise FileNotFoundError(
            "Tokenizer path not found. Place files under models/Qwen3-4B-Instruct-2507"
        )

    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)

    lengths: List[int] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = messages_to_text(obj.get("messages", []))
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))

    lengths.sort()
    n = len(lengths)
    print(f"samples: {n}")
    for p in [50, 75, 90, 95, 97, 98, 99, 99.5]:
        print(f"p{p}: {percentile(lengths, p)}")
    print(f"max: {lengths[-1] if lengths else 0}")


if __name__ == "__main__":
    main()
