#!/usr/bin/env python
"""
Strictly filter TOML-only samples.

Input: JSONL where each row has `messages` and TOML content in assistant turn.
Output: JSONL of samples that:
  - parse with tomllib
  - have no duplicate single-table headers
  - have no duplicate keys within a table instance

Usage:
  python scripts/filter_toml_dataset_strict.py \
    --input data/processed/20260211_v3/train_toml_only.jsonl \
    --output data/processed/20260211_v4/train_toml_only_strict.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import tomllib  # py3.11+
except Exception as e:  # pragma: no cover
    raise SystemExit("tomllib is required (Python 3.11+).") from e


HEADER_RE = re.compile(r"^\s*(\[\[?)([A-Za-z0-9_.-]+)(\]\]?)\s*$")
KEY_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*.+$")


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


def get_assistant_content(row: Dict) -> str:
    msgs = row.get("messages", [])
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def _strip_comment(line: str) -> str:
    # naive: strip from first unescaped '#'
    idx = line.find("#")
    if idx == -1:
        return line
    return line[:idx]


def _split_key_path(key: str) -> List[str]:
    return [p for p in key.split(".") if p]


def strict_toml_ok(text: str) -> bool:
    # Must parse.
    try:
        tomllib.loads(text)
    except Exception:
        return False

    seen_single_tables = set()
    current_table: Tuple[str, ...] = tuple()
    current_is_array = False
    keys_in_current = set()

    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue

        m = HEADER_RE.match(line)
        if m:
            left, name, right = m.groups()
            is_array = left == "[[" and right == "]]"
            table_path = tuple(_split_key_path(name))

            if not is_array:
                # single-table headers must be unique
                if table_path in seen_single_tables:
                    return False
                seen_single_tables.add(table_path)
                current_is_array = False
                current_table = table_path
                keys_in_current = set()
            else:
                # array-of-tables: allow repeated headers, but reset keys per element
                current_is_array = True
                current_table = table_path
                keys_in_current = set()
            continue

        km = KEY_RE.match(line)
        if km:
            key = km.group(1)
            key_path = tuple(current_table + tuple(_split_key_path(key)))
            # duplicate key in same table element is invalid
            if key_path in keys_in_current:
                return False
            keys_in_current.add(key_path)
            continue

        # If line is neither header nor key assignment, keep (tomllib already validated).
        # We don't reject here to avoid false positives.
        continue

    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict TOML filter for JSONL datasets.")
    ap.add_argument("--input", required=True, help="Input TOML-only JSONL")
    ap.add_argument("--output", required=True, help="Output filtered JSONL")
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
        content = get_assistant_content(row)
        if not content.strip():
            continue
        if strict_toml_ok(content):
            rows.append(row)
            kept += 1

    write_jsonl(out_path, rows)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Total: {total}")
    print(f"[INFO] Kept (strict): {kept}")


if __name__ == "__main__":
    main()
