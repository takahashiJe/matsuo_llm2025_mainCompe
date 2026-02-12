#!/usr/bin/env python
"""
Sanitize reasoning-inducing instructions in training prompts.

Conservative policy (default):
- Remove phrases like "Think step-by-step" only.
- Do NOT add new directives such as "Output only ...".
- Optionally restrict edits to TOML-target samples only.

Output:
- Writes *_sanitized.jsonl files into the specified output directory.
- Writes summary YAML with modification counts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import yaml


PATTERNS = [
    re.compile(r"\bthink step[- ]by[- ]step\b[, ]*", re.IGNORECASE),
    re.compile(r"\bstep[- ]by[- ]step\b[, ]*", re.IGNORECASE),
    re.compile(r"\bchain[- ]of[- ]thought\b[, ]*", re.IGNORECASE),
    re.compile(r"\breason(?:ing)? step[- ]by[- ]step\b[, ]*", re.IGNORECASE),
]


@dataclass
class Stats:
    total_rows: int = 0
    modified_rows: int = 0
    modified_system: int = 0
    modified_user: int = 0
    modified_assistant: int = 0


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


def sanitize_text(text: str, role: str) -> str:
    out = text
    for pat in PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = out.strip()
    return out


def infer_target_format(row: Dict) -> str:
    meta = row.get("metadata", {}) or {}
    tf = str(meta.get("target_format", "")).lower()
    if tf in ("json", "yaml", "toml", "xml", "csv"):
        return tf
    fmt = str(meta.get("format", "") or meta.get("output_type", "")).lower()
    if fmt in ("json", "yaml", "toml", "xml", "csv"):
        return fmt

    system = ""
    user = ""
    for m in row.get("messages", []):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system" and not system:
            system = str(m.get("content", ""))
        if m.get("role") == "user" and not user:
            user = str(m.get("content", ""))
    blob = f"{system}\n{user}".lower()
    for fmt_name in ("json", "yaml", "toml", "xml", "csv"):
        if f"expert in {fmt_name} format" in blob:
            return fmt_name
        if f"to {fmt_name}" in blob or f"into {fmt_name}" in blob:
            return fmt_name
    return "unknown"


def sanitize_file(src: Path, dst: Path, toml_only: bool) -> Stats:
    stats = Stats()
    out_rows: List[Dict] = []
    for row in iter_jsonl(src):
        stats.total_rows += 1
        row_changed = False
        new_row = json.loads(json.dumps(row))
        target_format = infer_target_format(new_row)
        if toml_only and target_format != "toml":
            out_rows.append(new_row)
            continue
        for m in new_row.get("messages", []):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            if role not in ("system", "user"):
                continue
            text = str(m.get("content", ""))
            sanitized = sanitize_text(text, role=role)
            if sanitized != text:
                m["content"] = sanitized
                row_changed = True
                if role == "system":
                    stats.modified_system += 1
                elif role == "user":
                    stats.modified_user += 1
                elif role == "assistant":
                    stats.modified_assistant += 1
        if row_changed:
            stats.modified_rows += 1
        out_rows.append(new_row)
    write_jsonl(dst, out_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize reasoning instructions from JSONL datasets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/20260212_v1/train_u10bei_denoised.jsonl",
            "data/processed/20260212_v1/phase2_mixed_denoised.jsonl",
            "data/processed/20260212_v1/train_toml_u10bei_daichira_valid_aligned.jsonl",
        ],
    )
    parser.add_argument("--out-dir", default="data/processed/20260212_v1")
    parser.add_argument(
        "--toml-only",
        action="store_true",
        help="If set, sanitize only TOML-target rows and leave other formats untouched.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    for src_str in args.inputs:
        src = Path(src_str)
        if not src.exists():
            summary[src_str] = {"exists": False}
            continue
        dst = out_dir / f"{src.stem}_sanitized.jsonl"
        st = sanitize_file(src, dst, toml_only=args.toml_only)
        summary[src_str] = {
            "exists": True,
            "output": str(dst),
            "stats": {
                "total_rows": st.total_rows,
                "modified_rows": st.modified_rows,
                "modified_system": st.modified_system,
                "modified_user": st.modified_user,
                "modified_assistant": st.modified_assistant,
            },
        }
        print(f"[DONE] {src} -> {dst} (modified_rows={st.modified_rows}/{st.total_rows})")

    summary_path = out_dir / "reasoning_sanitize_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
