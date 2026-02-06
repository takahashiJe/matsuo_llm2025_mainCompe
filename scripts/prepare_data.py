#!/usr/bin/env python
import argparse
from pathlib import Path

# Hugging Face Datasets: HF上のデータセットを簡単に取得・操作できるライブラリ
from datasets import load_dataset

DEFAULT_DATASETS = [
    "u-10bei/structured_data_with_cot_dataset_512_v2",
    "u-10bei/structured_data_with_cot_dataset_512_v4",
    "u-10bei/structured_data_with_cot_dataset_512_v5",
    "u-10bei/structured_data_with_cot_dataset_512",
    "u-10bei/structured_data_with_cot_dataset_v2",
    "u-10bei/structured_data_with_cot_dataset",
    "daichira/structured-3k-mix-sft",
    "daichira/structured-5k-mix-sft",
    "daichira/structured-hard-sft-4k"
]


def normalize_name(name: str) -> str:
    # ディレクトリ名に使えない "/" を "__" に置換して安全な名前にする
    return name.replace("/", "__")


def save_splits(ds_dict, out_dir: Path) -> None:
    # splitごとにJSONLで保存する（1行=1サンプル）
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in ds_dict.items():
        out_path = out_dir / f"{split}.jsonl"
        # HF Datasetsのto_jsonでJSONL形式に書き出し
        ds.to_json(out_path, orient="records", lines=True, force_ascii=False)


def main() -> None:
    # argparse: CLI引数を受け取るための標準ライブラリ
    parser = argparse.ArgumentParser(description="Download HF datasets and save as JSONL.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HF dataset names to download",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Output directory for raw data",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)

    for name in args.datasets:
        # load_dataset: 指定したHFデータセットを取得する
        print(f"Loading dataset: {name}")
        ds = load_dataset(name)
        dataset_dir = out_root / normalize_name(name)
        print(f"Saving to: {dataset_dir}")
        save_splits(ds, dataset_dir)


if __name__ == "__main__":
    main()
