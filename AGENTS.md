# AGENTS.md

このリポジトリで作業する際の共通ルールとセットアップ要点。

## 環境構築（uv + requirements.txt）
1. 仮想環境作成
```bash
uv venv .venv
```
2. 有効化
```bash
source .maLM25main/bin/activate
```
3. 依存インストール
```bash
uv pip install -r requirements.txt
```

## .env 管理（推奨）
`.env` に環境変数を置き、Git管理から除外する。

`.env` 例:
```bash
HF_HOME=./data/cache
HF_DATASETS_CACHE=./data/cache/datasets
```

読み込み（シェル方式）:
```bash
set -a
source .env
set +a
```

## ディレクトリ構成（最低限）
- `data/`
  - `raw/` 生データ
  - `processed/` 前処理済み
  - `splits/` train/valid/test
  - `cache/` HFキャッシュ
- `outputs/` 学習成果物

作成コマンド:
```bash
mkdir -p data/raw data/processed data/splits data/cache outputs
```

## .ipynb と .py の使い分け（探索→再現のサイクル）
- `notebooks/` は探索専用（理解・試行錯誤・小規模実験）
- `scripts/` は実行入口（再現性のある処理を回す）
- `src/` は再利用する本体ロジック（`scripts/` から呼び出す）
- うまくいった処理は `.ipynb` から `.py` に移植する

おすすめ構成:
```
notebooks/
  00_data_check.ipynb
  01_preprocess_try.ipynb
  02_train_sanity.ipynb
scripts/
  prepare_data.py
  train_lora.py
  evaluate.py
src/
  datasets/
    structured_dataset.py
  models/
    lora_utils.py
  training/
    train_loop.py
  utils/
    logging.py
    seed.py
```

### サイクルの頻度（目安）
- 初期（データ確認〜前処理）: 1〜2時間ごとに1サイクル
- 中期（学習パイプライン構築）: 1回の学習実験ごとに1サイクル
- 後期（改善・チューニング）: 変更1つにつき1サイクル
- 変更は一度に1つだけ入れる

## GPU確認
```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
PY
```

## 進め方（初期）
1. 小さなデータセットで中身確認（例: `daichira/structured-3k-mix-sft`）
2. 列名・サンプルを確認して前処理方針を決める
3. 前処理スクリプトを整備して統一フォーマットに変換
4. LoRA学習スクリプト作成 → 評価

## ハンズオン方針（理解しながら進める）
- すべてのステップは「打つコマンド」と「意味」をセットで確認する
- 機械学習コードも「目的・入力・出力・主要処理」を短く説明してから実装する
- コードは小さく書いて動かし、出力を見てから拡張する
- 1回の実行で出力を必ず観察し、次の仮説を立てる
- 失敗ログはメモし、再現できる状態で原因を切り分ける
- まず小さなデータ・短い学習で動作確認し、段階的に拡大する
- 変更は最小単位で行い、効果を確認してから次に進む

## 学習時のlossマスク方針（Outputタグ）
- 学習時は、assistant 出力内の **以下6タグのいずれかが最初に出現した位置の「直後」**から loss 計算を開始する。
  - `Output:`
  - `OUTPUT:`
  - `Final:`
  - `Answer:`
  - `Result:`
  - `Response:`
- 上記タグ以前のトークンは loss 対象外（`-100` マスク）とする。

## 重要な前提
- ベースモデル: `Qwen/Qwen3-4B-Instruct-2507`
- ベンチマーク: StructEval（StructEval-T）
- LLMによる合成データ作成は禁止（手動作成・コードでの改変はOK）

## コンペ要点（competition_summary.md の内容）
### 目的
- StructEval（StructEval-T）ベンチマークでの性能向上を競う
- 指定ベースモデルを用いた事後学習（LoRA等）が前提

### 指定ベースモデル
- `Qwen/Qwen3-4B-Instruct-2507`

### 使用可能データ（指定範囲）
- u-10bei 系
  - `u-10bei/structured_data_with_cot_dataset_512_v2`
  - `u-10bei/structured_data_with_cot_dataset_512_v4`
  - `u-10bei/structured_data_with_cot_dataset_512_v5`
  - `u-10bei/structured_data_with_cot_dataset_512`
  - `u-10bei/structured_data_with_cot_dataset_v2`
  - `u-10bei/structured_data_with_cot_dataset`
- daichira 系
  - `daichira/structured-3k-mix-sft`
  - `daichira/structured-5k-mix-sft`
  - `daichira/structured-hard-sft-4k`

### ルール/制約
- LLMを用いた合成データの作成は禁止
- 手動でのデータ作成は許可
- コードによるデータ改変は許可（例: ノイズ除去、整形、フィルタリング）
- 上記の指定データ範囲内で勝負

### 初参加向けの補足
- 事後学習の基本は「指定ベースモデル + 指定データ + LoRAなどの軽量学習」
- まずはデータを確認し、入出力フォーマットを揃えた学習パイプラインを作る
- 実験ログと評価指標（StructEval-T）を必ず記録する

## README.md ルール（運営指定・重要）
Hugging Face では README.md = モデルカード。第三者が再利用できる水準で
「何を学習し、どう使い、何に注意すべきか」を説明する義務がある。
README 不十分は「OSSとして不適切 / 学習内容が不透明 / ライセンス違反リスク」と評価される。

### 必須構成（この順で書く）
1. YAML メタデータ（必須）
```
---
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
- u-10bei/structured_data_with_cot_dataset_512_v2
language:
- en
license: Apache-2.0
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- lora
- structured-output
---
```
理由: HF 検索・分類・再現性に必須。無いと「壊れたモデルカード」扱い。

2. モデル概要（What）
```
# qwen3-4b-structured-sft-lora

This repository provides a **LoRA adapter** fine-tuned from
**Qwen3-4B-Instruct-2507** using **QLoRA (4-bit, Unsloth)**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.
```
必須ポイント: 「LoRAアダプタのみ」であること、ベースモデル名を明示。

3. 学習目的・設計思想（Why）
```
## Training Objective

This adapter is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.
```
重要: assistant-only loss, CoT mask（Output: 以降のみ学習）。

4. 学習設定（How）
```
## Training Configuration

- Base model: Qwen3-4B-Instruct-2507
- Method: QLoRA (4-bit)
- Max sequence length: 512
- Epochs: 1
- Learning rate: 1e-6
- LoRA: r=64, alpha=128
```
再現性のため必須。

5. 使用方法（How to use）
```
## Usage

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "JuntaTakahashi/qwen3-4b-structured-sft-lora"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

6. データセット・ライセンス注意（必須・重要）
```
## Sources & Terms (IMPORTANT)

Training data: u-10bei/structured_data_with_cot_dataset_512_v2

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
```

### 実行ルール
- 最低限、モデルタイトルの欄は必ず自分で書き込む。
- 使用データセットを変更した場合は、`Dataset License` と `Compliance` を適切に書き換える。

### 今回のモデル情報
- モデルタイトル: `qwen3-4b-structured-sft-lora`
- リポジトリ: `JuntaTakahashi/qwen3-4b-structured-sft-lora`
