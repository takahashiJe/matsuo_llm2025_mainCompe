#!/usr/bin/env python
"""
Strictest TOML filter: reject any duplicate table headers or key paths.

This enforces:
  - tomllib parseable
  - no repeated table headers (including array-of-tables)
  - no repeated key paths anywhere in the document

Usage:
  python scripts/filter_toml_dataset_no_dupes.py \
    --input data/processed/20260211_v4/train_toml_only_parsed.jsonl \
    --output data/processed/20260211_v5/train_toml_only_nodup.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

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
    idx = line.find("#")
    if idx == -1:
        return line
    return line[:idx]


def _split_key_path(key: str) -> List[str]:
    return [p for p in key.split(".") if p]


def no_duplicate_toml(text: str) -> bool:
    # Must parse.
    try:
        tomllib.loads(text)
    except Exception:
        return False

    seen_tables = set()
    seen_keys = set()
    current_table: Tuple[str, ...] = tuple()

    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue

        m = HEADER_RE.match(line)
        if m:
            _, name, _ = m.groups()
            table_path = tuple(_split_key_path(name))
            # Disallow any repeated table header (including array-of-tables)
            if table_path in seen_tables:
                return False
            seen_tables.add(table_path)
            current_table = table_path
            continue

        km = KEY_RE.match(line)
        if km:
            key = km.group(1)
            key_path = tuple(current_table + tuple(_split_key_path(key)))
            if key_path in seen_keys:
                return False
            seen_keys.add(key_path)
            continue

        # If line is neither header nor key assignment, keep (tomllib validated).
        continue

    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter TOML JSONL to zero-duplicate keys/tables.")
    ap.add_argument("--input", required=True, help="Input TOML JSONL (already TOML-only)")
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
        if no_duplicate_toml(content):
            rows.append(row)
            kept += 1

    write_jsonl(out_path, rows)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Total: {total}")
    print(f"[INFO] Kept (no-dup): {kept}")


if __name__ == "__main__":
    main()
