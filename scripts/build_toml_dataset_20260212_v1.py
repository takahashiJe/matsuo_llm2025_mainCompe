#!/usr/bin/env python
"""
Build TOML-focused training datasets from u10bei + daichira.

Rules:
- Keep samples only when TOML is clearly the target format.
- Validate assistant output with tomllib.
- Write per-source outputs and merged deduplicated output.

Usage:
  python scripts/build_toml_dataset_20260212_v1.py \
    --u10bei data/processed/20260209_v1/train_u10bei.jsonl \
    --daichira data/processed/20260209_v1/train_daichira.jsonl \
    --out-dir data/processed/20260212_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import yaml

try:
    import tomllib  # py3.11+
except Exception as exc:  # pragma: no cover
    raise SystemExit("tomllib is required (Python 3.11+).") from exc


TARGET_PATTERNS = [
    re.compile(r"\bto toml\b", re.IGNORECASE),
    re.compile(r"\binto toml\b", re.IGNORECASE),
    re.compile(r"\boutput toml\b", re.IGNORECASE),
    re.compile(r"\btoml format\b", re.IGNORECASE),
    re.compile(r"\btoml structure\b", re.IGNORECASE),
    re.compile(r"\btoml representation\b", re.IGNORECASE),
]


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


def get_turn_text(row: Dict, role: str) -> str:
    for m in row.get("messages", []):
        if isinstance(m, dict) and m.get("role") == role:
            return str(m.get("content", ""))
    return ""


def is_toml_target(row: Dict) -> bool:
    meta = row.get("metadata", {}) or {}
    fmt = str(meta.get("format", "") or meta.get("output_type", "")).lower()
    if fmt == "toml":
        return True

    system = get_turn_text(row, "system")
    user = get_turn_text(row, "user")
    blob = f"{system}\n{user}".lower()
    if "expert in toml format" in blob:
        return True
    return any(p.search(blob) is not None for p in TARGET_PATTERNS)


def is_valid_toml_assistant(row: Dict) -> bool:
    assistant = get_turn_text(row, "assistant").strip()
    if not assistant:
        return False
    try:
        tomllib.loads(assistant)
        return True
    except Exception:
        return False


def sample_signature(row: Dict) -> str:
    user = get_turn_text(row, "user").strip()
    assistant = get_turn_text(row, "assistant").strip()
    key = f"{user}\n---\n{assistant}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def annotate_row(row: Dict, source: str) -> Dict:
    out = dict(row)
    meta = dict(out.get("metadata", {}) or {})
    meta["source_dataset"] = source
    out["metadata"] = meta
    return out


def extract_source(path: Path, source_name: str) -> Tuple[List[Dict], Dict[str, int]]:
    total = 0
    target = 0
    valid = 0
    rows: List[Dict] = []
    for row in read_jsonl(path):
        total += 1
        if not is_toml_target(row):
            continue
        target += 1
        if not is_valid_toml_assistant(row):
            continue
        valid += 1
        rows.append(annotate_row(row, source_name))
    return rows, {"total": total, "toml_target": target, "toml_valid": valid}


def merge_dedup(rows_a: List[Dict], rows_b: List[Dict]) -> Tuple[List[Dict], Counter]:
    seen = set()
    dup_stats = Counter()
    merged: List[Dict] = []
    for row in rows_a + rows_b:
        sig = sample_signature(row)
        if sig in seen:
            src = str((row.get("metadata", {}) or {}).get("source_dataset", "unknown"))
            dup_stats[src] += 1
            continue
        seen.add(sig)
        merged.append(row)
    return merged, dup_stats


def to_schema_aligned_rows(rows: List[Dict]) -> List[Dict]:
    """
    Convert heterogeneous records into a stable schema for datasets library:
    - top-level: messages, metadata
    - metadata keys are fixed across rows
    """
    aligned: List[Dict] = []
    for row in rows:
        messages = row.get("messages", [])
        metadata = row.get("metadata", {}) or {}
        source = str(metadata.get("source_dataset", "unknown"))
        out = {
            "messages": messages,
            "metadata": {
                "source_dataset": source,
                "target_format": "toml",
                "loss_mask_mode": str(metadata.get("loss_mask_mode", "auto")),
            },
        }
        aligned.append(out)
    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TOML dataset from u10bei + daichira.")
    ap.add_argument("--u10bei", required=True, help="Path to u10bei train jsonl")
    ap.add_argument("--daichira", required=True, help="Path to daichira train jsonl")
    ap.add_argument("--out-dir", default="data/processed/20260212_v1", help="Output directory")
    args = ap.parse_args()

    u10bei_path = Path(args.u10bei)
    daichira_path = Path(args.daichira)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not u10bei_path.exists():
        raise FileNotFoundError(f"u10bei input not found: {u10bei_path}")
    if not daichira_path.exists():
        raise FileNotFoundError(f"daichira input not found: {daichira_path}")

    u_rows, u_stats = extract_source(u10bei_path, "u10bei")
    d_rows, d_stats = extract_source(daichira_path, "daichira")
    merged_rows, dup_stats = merge_dedup(u_rows, d_rows)
    merged_aligned_rows = to_schema_aligned_rows(merged_rows)

    u_out = out_dir / "train_toml_u10bei_valid.jsonl"
    d_out = out_dir / "train_toml_daichira_valid.jsonl"
    m_out = out_dir / "train_toml_u10bei_daichira_valid.jsonl"
    m_aligned_out = out_dir / "train_toml_u10bei_daichira_valid_aligned.jsonl"

    write_jsonl(u_out, u_rows)
    write_jsonl(d_out, d_rows)
    write_jsonl(m_out, merged_rows)
    write_jsonl(m_aligned_out, merged_aligned_rows)

    summary = {
        "inputs": {"u10bei": str(u10bei_path), "daichira": str(daichira_path)},
        "outputs": {
            "u10bei": str(u_out),
            "daichira": str(d_out),
            "merged": str(m_out),
            "merged_aligned": str(m_aligned_out),
        },
        "stats": {
            "u10bei": u_stats,
            "daichira": d_stats,
            "merged_before_dedup": len(u_rows) + len(d_rows),
            "merged_after_dedup": len(merged_rows),
            "merged_aligned_rows": len(merged_aligned_rows),
            "dedup_dropped": dict(dup_stats),
        },
    }
    (out_dir / "build_summary.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print("[DONE] TOML datasets created.")
    print(f"  - {u_out} ({len(u_rows)})")
    print(f"  - {d_out} ({len(d_rows)})")
    print(f"  - {m_out} ({len(merged_rows)})")
    print(f"  - {m_aligned_out} ({len(merged_aligned_rows)})")
    print(f"  - {out_dir / 'build_summary.yaml'}")


if __name__ == "__main__":
    main()
