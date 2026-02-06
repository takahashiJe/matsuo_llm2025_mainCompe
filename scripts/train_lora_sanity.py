#!/usr/bin/env python
"""
train_lora.py と同一ロジックを使う Sanity 実行用エントリポイント。
"""

from __future__ import annotations

import argparse

from train_lora import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL QLoRA trainer (sanity mode).")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data.data_path from config",
    )
    parser.add_argument(
        "--second-data-path",
        type=str,
        default=None,
        help="Optional second stage data path; trained after --data-path",
    )
    parser.add_argument("--sanity-train-size", type=int, default=64)
    parser.add_argument("--sanity-eval-size", type=int, default=16)
    parser.add_argument(
        "--sanity-epochs",
        type=float,
        default=None,
        help="Override num_train_epochs only for sanity run",
    )
    parser.add_argument(
        "--sanity-logging-steps",
        type=int,
        default=None,
        help="Override logging_steps only for sanity run",
    )
    parser.add_argument("--sanity-prefix", type=str, default="sanity-")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        data_path_override=args.data_path,
        second_data_path_override=args.second_data_path,
        sanity=True,
        sanity_train_size=args.sanity_train_size,
        sanity_eval_size=args.sanity_eval_size,
        sanity_run_name_prefix=args.sanity_prefix,
        sanity_num_train_epochs=args.sanity_epochs,
        sanity_logging_steps=args.sanity_logging_steps,
    )


if __name__ == "__main__":
    main()
