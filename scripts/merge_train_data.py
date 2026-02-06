#!/usr/bin/env python
"""
学習データを整形・重複除去し、u-10bei系 / daichira系を別々に出力する。

方針:
- u-10bei: v5 を主軸 + v2 を追加 (prompt+output 重複を削除)
- daichira: 3k + 5k + hard を追加 (prompt+output 重複を削除)

出力:
- `messages` + `metadata` の統一スキーマ JSONL
- `data/processed/train_u10bei.jsonl`
- `data/processed/train_daichira.jsonl`
- （任意）`data/processed/merged_train.jsonl`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# -----------------------------
# 設定（方針に合わせて固定）
# -----------------------------

DEFAULT_U10BEI_SOURCES = [
    "data/raw/u-10bei__structured_data_with_cot_dataset_512_v5/train.jsonl",
    "data/raw/u-10bei__structured_data_with_cot_dataset_v2/train.jsonl",
]

DEFAULT_DAICHIRA_SOURCES = [
    "data/raw/daichira__structured-3k-mix-sft/train.jsonl",
    "data/raw/daichira__structured-5k-mix-sft/train.jsonl",
    "data/raw/daichira__structured-hard-sft-4k/train.jsonl",
]

METADATA_FIELDS = [
    "format",
    "complexity",
    "schema",
    "constraint",
    "type",
    "prompt",
    "output",
    "estimated_tokens",
    "source",
    "id",
    "category",
    "subcategory",
    "task",
    "seed",
]

META_DEFAULTS: Dict[str, object] = {
    "format": "",
    "complexity": "",
    "schema": "",
    "constraint": "",
    "type": "",
    "prompt": "",
    "output": "",
    "estimated_tokens": -1,
    "source": "",
    "id": "",
    "category": "",
    "subcategory": "",
    "task": "",
    "seed": "",
}


# -----------------------------
# 低レベル: JSON 読み込み
# -----------------------------

def iter_json_objects(text: str) -> Iterator[dict]:
    """
    JSONL / 連続JSON / pretty-printed など、
    1ファイル内に複数JSONが連結されている形式を安全に読む。
    """
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        # 空白を飛ばす
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        yield obj
        i = j


def load_records(path: Path) -> List[dict]:
    """
    ファイルからレコードを読み込む。
    - JSONL (1行1件)
    - 連続JSON (daichira のような複数 JSON が直列)
    のどちらにも対応する。
    """
    text = path.read_text(encoding="utf-8")
    # まずは raw_decode でまとめて読む
    try:
        return list(iter_json_objects(text))
    except Exception:
        # 念のため行単位の fallback も用意
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


# -----------------------------
# スキーマ正規化
# -----------------------------

def norm_text(text: Optional[str]) -> Optional[str]:
    """重複判定のために空白を正規化する。"""
    if text is None:
        return None
    return " ".join(text.strip().split())


def extract_prompt_output(record: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    prompt / output を取り出す。
    - u-10bei: metadata.prompt / metadata.output が優先
    - それ以外: messages から user / assistant を拾う
    """
    meta = record.get("metadata")
    if isinstance(meta, dict):
        prompt = meta.get("prompt")
        output = meta.get("output")
        if prompt is not None and output is not None:
            return prompt, output

    prompt = None
    output = None
    for m in record.get("messages", []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user" and prompt is None:
            prompt = m.get("content")
        if role == "assistant" and output is None:
            output = m.get("content")
    return prompt, output


def normalize_record(record: dict, source: str) -> dict:
    """
    `messages + metadata` の統一スキーマに変換する。
    - u-10bei: metadata を引き継ぎ、source を追加
    - daichira: id/category/... を metadata に移す
      かつ assistant 先頭に `Output: ` を付与（Output 部分を明示）
    """
    messages = record.get("messages", [])
    meta: Dict[str, object] = dict(META_DEFAULTS)
    prompt, output = extract_prompt_output(record)
    meta["prompt"] = prompt if isinstance(prompt, str) else ""
    meta["output"] = output if isinstance(output, str) else ""

    if "metadata" in record and isinstance(record["metadata"], dict):
        # u-10bei 系
        for key in ["format", "complexity", "schema", "constraint", "type", "estimated_tokens"]:
            if key in record["metadata"]:
                value = record["metadata"][key]
                if value is not None:
                    meta[key] = value
    else:
        # daichira 系
        for key in ["id", "category", "subcategory", "task", "seed"]:
            if key in record:
                meta[key] = record[key]
        # daichira 系は Output タグが無いので付与する
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, str) and not content.lstrip().startswith("Output:"):
                    m["content"] = "Output: " + content
                break

    meta["source"] = source

    return {
        "messages": messages,
        "metadata": meta,
    }


# -----------------------------
# メイン処理
# -----------------------------

@dataclass
class Stats:
    read: int = 0
    kept: int = 0
    dropped_dup: int = 0
    dropped_missing: int = 0


def merge_and_dedupe(paths: List[Path]) -> Tuple[List[dict], Dict[str, Stats]]:
    """
    `prompt + output` で重複除去しつつ、全レコードを統合する。
    """
    merged: List[dict] = []
    seen_keys = set()
    stats: Dict[str, Stats] = {}

    for path in paths:
        source = path.parent.name  # ディレクトリ名を source として使う
        stats[source] = Stats()
        records = load_records(path)
        stats[source].read = len(records)

        for record in records:
            prompt, output = extract_prompt_output(record)
            prompt_n = norm_text(prompt)
            output_n = norm_text(output)

            # prompt / output が欠けているものは除外
            if prompt_n is None or output_n is None:
                stats[source].dropped_missing += 1
                continue

            key = (prompt_n, output_n)
            if key in seen_keys:
                stats[source].dropped_dup += 1
                continue

            seen_keys.add(key)
            merged.append(normalize_record(record, source=source))
            stats[source].kept += 1

    return merged, stats


def write_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_report(
    u_stats: Dict[str, Stats],
    daichira_stats: Dict[str, Stats],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Merge Report", ""]
    lines.append("## u-10bei")
    lines.append("")
    for source, s in u_stats.items():
        lines.append(f"## {source}")
        lines.append(f"- read: {s.read}")
        lines.append(f"- kept: {s.kept}")
        lines.append(f"- dropped_dup: {s.dropped_dup}")
        lines.append(f"- dropped_missing: {s.dropped_missing}")
        lines.append("")
    lines.append("## daichira")
    lines.append("")
    for source, s in daichira_stats.items():
        lines.append(f"## {source}")
        lines.append(f"- read: {s.read}")
        lines.append(f"- kept: {s.kept}")
        lines.append(f"- dropped_dup: {s.dropped_dup}")
        lines.append(f"- dropped_missing: {s.dropped_missing}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train datasets with prompt+output dedupe (u-10bei / daichira split)."
    )
    parser.add_argument(
        "--u10bei-sources",
        nargs="+",
        default=DEFAULT_U10BEI_SOURCES,
        help="Input train.jsonl paths for u-10bei family",
    )
    parser.add_argument(
        "--daichira-sources",
        nargs="+",
        default=DEFAULT_DAICHIRA_SOURCES,
        help="Input train.jsonl paths for daichira family",
    )
    parser.add_argument(
        "--out-u10bei",
        default="data/processed/train_u10bei.jsonl",
        help="Output JSONL path for u-10bei family",
    )
    parser.add_argument(
        "--out-daichira",
        default="data/processed/train_daichira.jsonl",
        help="Output JSONL path for daichira family",
    )
    parser.add_argument(
        "--out-merged",
        default="",
        help="Optional merged output JSONL path (set empty string to skip)",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/merge_report.md",
        help="Report output path",
    )
    args = parser.parse_args()

    u_paths = [Path(p) for p in args.u10bei_sources]
    d_paths = [Path(p) for p in args.daichira_sources]
    u_records, u_stats = merge_and_dedupe(u_paths)
    d_records, d_stats = merge_and_dedupe(d_paths)

    write_jsonl(u_records, Path(args.out_u10bei))
    write_jsonl(d_records, Path(args.out_daichira))

    if args.out_merged.strip():
        merged = u_records + d_records
        write_jsonl(merged, Path(args.out_merged))

    write_report(u_stats, d_stats, Path(args.report))

    u_total = sum(s.kept for s in u_stats.values())
    d_total = sum(s.kept for s in d_stats.values())
    print(f"Saved u-10bei: {args.out_u10bei} ({u_total} records)")
    print(f"Saved daichira: {args.out_daichira} ({d_total} records)")
    if args.out_merged.strip():
        print(f"Saved merged: {args.out_merged} ({u_total + d_total} records)")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
