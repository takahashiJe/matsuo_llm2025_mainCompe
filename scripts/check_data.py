#!/usr/bin/env python
"""
9つの指定データセットに対する基本的なデータ品質チェック。

チェック対象:
- 列名の一致
- 重複行
- 欠損値
"""

from __future__ import annotations

# 標準ライブラリ
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Any, Iterable, Mapping, Optional

# Hugging Face Datasets: HF Hubのデータセットを読み込み・操作するライブラリ
from datasets import Dataset, DatasetDict, load_dataset


# 1) データセット名をグローバルに定義（要望に合わせる）
DATASET_NAMES: List[str] = [
    "u-10bei/structured_data_with_cot_dataset_512_v2",
    "u-10bei/structured_data_with_cot_dataset_512_v4",
    "u-10bei/structured_data_with_cot_dataset_512_v5",
    "u-10bei/structured_data_with_cot_dataset_512",
    "u-10bei/structured_data_with_cot_dataset_v2",
    "u-10bei/structured_data_with_cot_dataset",
    "daichira/structured-3k-mix-sft",
    "daichira/structured-5k-mix-sft",
    "daichira/structured-hard-sft-4k",
]


# 2) データセットをグローバルで読み込み（要望に合わせる）
#    NOTE: import時にダウンロードが走る。
#    遅い場合はmain()内に移動する。
DATASETS: Dict[str, DatasetDict] = {name: load_dataset(name) for name in DATASET_NAMES}


def iter_json_objects(text: str) -> Iterable[dict]:
    """
    JSONL / 連続JSON / pretty-printed など、
    1ファイル内に複数JSONが連結されている形式を安全に読む。
    """
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        yield obj
        i = j


class SimpleDataset:
    """
    HF Datasets を使わずに、必要最低限の操作だけ提供する軽量ラッパー。
    - column_names: 全レコードのキーの合算
    - ds[col]: そのカラムの値のリスト（欠損は None）
    """

    def __init__(self, records: List[dict]) -> None:
        self._records = records
        columns = []
        seen = set()
        for rec in records:
            for key in rec.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        self._columns = columns

    @property
    def column_names(self) -> List[str]:
        return self._columns

    def __getitem__(self, key: str) -> List[object]:
        values: List[object] = []
        for rec in self._records:
            values.append(rec.get(key))
        return values


def load_local_jsonl(path: Path) -> Dict[str, Dict[str, SimpleDataset]]:
    """
    JSONLをローカルから読み込み、軽量ラッパーで扱う。
    - merged_train.jsonl のようにsplitがない場合は train のみで扱う。
    """
    text = path.read_text(encoding="utf-8")
    records = list(iter_json_objects(text))
    return {path.stem: {"train": SimpleDataset(records)}}


def check_column_consistency(
    datasets: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """列名がデータセット間・split間で一致しているかを確認する。"""
    print("\n[Check] Column consistency")

    # データセットごと・splitごとに列名を集める
    columns_map = {}
    for name, ds_dict in datasets.items():
        columns_map[name] = {split: ds.column_names for split, ds in ds_dict.items()}

    # 最初のデータセットの最初のsplitを基準に比較する
    ref_name = next(iter(columns_map))
    ref_split = next(iter(columns_map[ref_name]))
    ref_columns = columns_map[ref_name][ref_split]

    results: Dict[str, Dict[str, Any]] = {}
    for name, splits in columns_map.items():
        results[name] = {"reference": ref_columns, "splits": {}}
        for split, cols in splits.items():
            if cols != ref_columns:
                print(f"- MISMATCH: {name} [{split}] columns={cols}")
                results[name]["splits"][split] = {"match": False, "columns": cols}
            else:
                print(f"- OK: {name} [{split}] columns match reference")
                results[name]["splits"][split] = {"match": True, "columns": cols}

    return results


def check_duplicates(
    datasets: Mapping[str, Mapping[str, Any]],
    key: str = "messages",
) -> Dict[str, Dict[str, Any]]:
    """
    指定カラムを基準に重複行をチェックする。
    - key: 重複判定に使うカラム（デフォルト: 'messages'）
    """
    print("\n[Check] Duplicates by key")

    results: Dict[str, Dict[str, Any]] = {}
    for name, ds_dict in datasets.items():
        results[name] = {}
        for split, ds in ds_dict.items():
            if key not in ds.column_names:
                print(f"- SKIP: {name} [{split}] missing key '{key}'")
                results[name][split] = {"skipped": True, "duplicates": None}
                continue

            # 文字列化して比較可能にする
            values = [str(x) for x in ds[key]]
            counts = Counter(values)
            dup_count = sum(1 for v in counts.values() if v > 1)
            print(f"- {name} [{split}] duplicates={dup_count}")
            results[name][split] = {"skipped": False, "duplicates": dup_count}

    return results


def check_missing_values(
    datasets: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """各カラムの欠損値（Noneまたは空文字）を数える。"""
    print("\n[Check] Missing values")

    results: Dict[str, Dict[str, Any]] = {}
    for name, ds_dict in datasets.items():
        results[name] = {}
        for split, ds in ds_dict.items():
            missing_stats = {}
            for col in ds.column_names:
                values = ds[col]
                missing = sum(1 for v in values if v is None or v == "")
                missing_stats[col] = missing

            print(f"- {name} [{split}] missing={missing_stats}")
            results[name][split] = missing_stats

    return results


def save_reports(
    out_dir: Path,
    column_report: Dict[str, Dict[str, Any]],
    duplicate_report: Dict[str, Dict[str, Any]],
    missing_report: Dict[str, Dict[str, Any]],
    dataset_names: List[str],
) -> None:
    """JSONとMarkdownでチェック結果を保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON出力（機械処理用）
    json_path = out_dir / "check_data.json"
    payload = {
        "columns": column_report,
        "duplicates": duplicate_report,
        "missing": missing_report,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown出力（人間が読む用）
    md_path = out_dir / "check_data.md"
    lines = ["# Data Check Report", ""]
    for name in dataset_names:
        lines.append(f"## {name}")
        lines.append("- columns:")
        for split, info in column_report[name]["splits"].items():
            status = "OK" if info["match"] else "MISMATCH"
            lines.append(f"  - {split}: {status}")
        lines.append("- duplicates:")
        for split, info in duplicate_report[name].items():
            if info["skipped"]:
                lines.append(f"  - {split}: SKIP (missing key)")
            else:
                lines.append(f"  - {split}: {info['duplicates']}")
        lines.append("- missing:")
        for split, stats in missing_report[name].items():
            lines.append(f"  - {split}: {stats}")
        lines.append("- 特徴:")
        lines.append("  - TODO: 人間が観察した特徴を記入")
        lines.append("- 修正案:")
        lines.append("  - TODO: どのような前処理を入れるか記入")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # argparse: CLI引数を受け取るための標準ライブラリ
    import argparse

    parser = argparse.ArgumentParser(description="Check dataset quality.")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="ローカルJSONLを指定（例: data/processed/merged_train.jsonl）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reports",
        help="レポート出力先ディレクトリ",
    )
    args = parser.parse_args()

    if args.jsonl:
        datasets = load_local_jsonl(Path(args.jsonl))
        print("Datasets loaded (local jsonl):", list(datasets.keys()))
    else:
        datasets = DATASETS
        print("Datasets loaded:", list(datasets.keys()))

    column_report = check_column_consistency(datasets)
    duplicate_report = check_duplicates(datasets)
    missing_report = check_missing_values(datasets)

    save_reports(
        Path(args.out_dir),
        column_report,
        duplicate_report,
        missing_report,
        list(datasets.keys()),
    )
    print(f"\nSaved reports to {args.out_dir}/")


if __name__ == "__main__":
    main()
