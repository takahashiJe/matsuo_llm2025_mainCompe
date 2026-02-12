#!/usr/bin/env python
"""
Denoise phase1/phase2 training datasets without using benchmark task patterns.

Policy:
- Keep only samples with valid message structure (user + assistant present).
- Infer target format from metadata/system/user text when possible.
- Validate assistant output syntax for expected format.
- If target format is unknown, keep only samples parseable as one of JSON/YAML/TOML/XML/CSV.
- Drop exact duplicates by normalized (system, user, assistant) tuple.

Outputs default to: data/processed/20260212_v1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import yaml

try:
    import tomllib  # py3.11+
except Exception as exc:  # pragma: no cover
    raise SystemExit("tomllib is required (Python 3.11+).") from exc


FORMATS = ("json", "yaml", "toml", "xml", "csv")


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_missing_turns: int = 0
    dropped_empty_content: int = 0
    dropped_invalid_expected_format: int = 0
    dropped_unparseable_unknown_format: int = 0
    dropped_duplicates: int = 0


def iter_jsonl(path: Path) -> Iterator[Dict]:
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


def get_turns(row: Dict) -> Tuple[str, str, str]:
    system = ""
    user = ""
    assistant = ""
    for m in row.get("messages", []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "system" and not system:
            system = content
        elif role == "user" and not user:
            user = content
        elif role == "assistant" and not assistant:
            assistant = content
    return system, user, assistant


def norm_text(text: str) -> str:
    return " ".join(text.strip().split())


def parse_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def parse_yaml(text: str) -> bool:
    try:
        obj = yaml.safe_load(text)
        return isinstance(obj, (dict, list))
    except Exception:
        return False


def parse_toml(text: str) -> bool:
    try:
        tomllib.loads(text)
        return True
    except Exception:
        return False


def parse_xml(text: str) -> bool:
    try:
        ET.fromstring(text)
        return True
    except Exception:
        return False


def parse_csv(text: str) -> bool:
    try:
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 2:
            return False
        width = len(rows[0])
        if width < 2:
            return False
        return all(len(r) == width for r in rows[: min(5, len(rows))])
    except Exception:
        return False


def is_valid_for_format(fmt: str, text: str) -> bool:
    if fmt == "json":
        return parse_json(text)
    if fmt == "yaml":
        return parse_yaml(text)
    if fmt == "toml":
        return parse_toml(text)
    if fmt == "xml":
        return parse_xml(text)
    if fmt == "csv":
        return parse_csv(text)
    return False


def detect_any_format(text: str) -> Optional[str]:
    for fmt in FORMATS:
        if is_valid_for_format(fmt, text):
            return fmt
    return None


def infer_expected_format(row: Dict, system: str, user: str) -> Optional[str]:
    metadata = row.get("metadata", {}) or {}
    meta_fmt = str(metadata.get("format", "") or metadata.get("output_type", "")).strip().lower()
    if meta_fmt in FORMATS:
        return meta_fmt

    sys_low = system.lower()
    m = re.search(r"expert in\s+(json|yaml|toml|xml|csv)\s+format", sys_low)
    if m:
        return m.group(1)

    user_low = user.lower()
    for fmt in FORMATS:
        if f"to {fmt}" in user_low or f"into {fmt}" in user_low:
            return fmt
        if f"output {fmt}" in user_low or f"{fmt} format" in user_low:
            return fmt
    return None


def filter_rows(rows: Iterator[Dict]) -> Tuple[list[Dict], FilterStats, Counter]:
    out: list[Dict] = []
    stats = FilterStats()
    kept_formats: Counter = Counter()
    seen = set()

    for row in rows:
        stats.total += 1
        system, user, assistant = get_turns(row)
        if not user or not assistant:
            stats.dropped_missing_turns += 1
            continue
        if not user.strip() or not assistant.strip():
            stats.dropped_empty_content += 1
            continue

        expected = infer_expected_format(row, system, user)
        if expected is not None:
            if not is_valid_for_format(expected, assistant):
                stats.dropped_invalid_expected_format += 1
                continue
            valid_fmt = expected
        else:
            valid_fmt = detect_any_format(assistant)
            if valid_fmt is None:
                stats.dropped_unparseable_unknown_format += 1
                continue

        dedupe_key = (
            norm_text(system),
            norm_text(user),
            norm_text(assistant),
        )
        if dedupe_key in seen:
            stats.dropped_duplicates += 1
            continue
        seen.add(dedupe_key)

        out.append(row)
        kept_formats[valid_fmt] += 1
        stats.kept += 1

    return out, stats, kept_formats


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise phase1/phase2 training datasets.")
    parser.add_argument("--phase1", default="data/processed/20260209_v1/train_daichira.jsonl")
    parser.add_argument("--phase2-u10bei", default="data/processed/20260209_v1/train_u10bei.jsonl")
    parser.add_argument("--phase2-mixed", default="outputs/models/20260211_v5/phase2_mixed.jsonl")
    parser.add_argument("--out-dir", default="data/processed/20260212_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "phase1_daichira": Path(args.phase1),
        "phase2_u10bei": Path(args.phase2_u10bei),
        "phase2_mixed": Path(args.phase2_mixed),
    }

    summary: Dict[str, Dict] = {}
    for name, path in inputs.items():
        if not path.exists():
            summary[name] = {
                "input": str(path),
                "exists": False,
            }
            continue

        cleaned, stats, kept_formats = filter_rows(iter_jsonl(path))
        out_path = out_dir / f"{path.stem}_denoised.jsonl"
        write_jsonl(out_path, cleaned)

        summary[name] = {
            "input": str(path),
            "exists": True,
            "output": str(out_path),
            "stats": {
                "total": stats.total,
                "kept": stats.kept,
                "dropped_missing_turns": stats.dropped_missing_turns,
                "dropped_empty_content": stats.dropped_empty_content,
                "dropped_invalid_expected_format": stats.dropped_invalid_expected_format,
                "dropped_unparseable_unknown_format": stats.dropped_unparseable_unknown_format,
                "dropped_duplicates": stats.dropped_duplicates,
            },
            "kept_formats": dict(kept_formats),
        }

    summary_path = out_dir / "phase12_denoise_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"[DONE] wrote summary: {summary_path}")
    for key, info in summary.items():
        if not info.get("exists"):
            print(f"- {key}: missing input")
            continue
        print(f"- {key}: kept {info['stats']['kept']} / {info['stats']['total']} -> {info['output']}")


if __name__ == "__main__":
    main()

