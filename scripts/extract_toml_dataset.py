#!/usr/bin/env python
"""
Extract TOML-only samples from a JSONL dataset.

Expected input format: each line is a JSON object with at least:
- messages: list of chat turns
- metadata.format: should be "toml" for TOML samples

Usage:
  python scripts/extract_toml_dataset.py \
    --input data/processed/20260209_v1/train_u10bei.jsonl \
    --output data/processed/20260211_v3/train_toml_only.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator


def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def is_toml_sample(row: Dict) -> bool:
    meta = row.get("metadata", {})
    fmt = meta.get("format") or meta.get("output_type")
    if isinstance(fmt, str) and fmt.lower() == "toml":
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract TOML-only samples from JSONL.")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    total = 0
    kept = 0
    rows = []
    for row in read_jsonl(in_path):
        total += 1
        if is_toml_sample(row):
            rows.append(row)
            kept += 1

    write_jsonl(out_path, rows)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Total: {total}")
    print(f"[INFO] TOML kept: {kept}")


if __name__ == "__main__":
    main()
