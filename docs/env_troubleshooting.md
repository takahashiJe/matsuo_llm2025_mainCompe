# 環境/依存関係トラブルシュート

ここでは「なぜ失敗したのか」と「どう直せばいいか」を、できるだけ簡単にまとめます。

## 1) `uv pip install` が依存解決できない

### 症状（エラーの意味）
`datasets==4.3.0` が **新しい `requests`** を必要としているのに、`uv` が **古い `requests`** しか見つけられない、というエラー。

### 何が原因か
`uv` は「最初に見つけた配布元（index）」だけを見るルールがあります。  
今回、PyTorch の配布元で古い `requests` が見つかったため、PyPI の新しい `requests` を見に行きませんでした。

### どう直すか（1行でOK）
コマンドに `--index-strategy unsafe-best-match` を付けます。  
これで **PyPI の新しい `requests`** も探してくれるようになります。

```bash
uv pip install --index-strategy unsafe-best-match -r requirements.txt
```

### 毎回付けたくない場合（任意）
`.env` に追加して固定できます。

```bash
UV_INDEX_STRATEGY=unsafe-best-match
```

読み込み例:
```bash
set -a
source .env
set +a
```

---

## 2) `ImportError: cannot import name 'importlib_version' ...` が出る

### 症状
`from unsloth import FastLanguageModel` でエラーになる。

### まず確認すること
Jupyter が **正しい Python 環境** を使っているか確認します。

```python
import sys
print(sys.executable)
```

期待される表示:
`/home/junta_takahashi/matsuo_llm2025_mainCompe/.maLM25main/bin/python`

### どう直すか（実際に効いた方法）
**カーネル再起動 → 再インポート** で直りました。

まだ直らない場合は、**読み込み済みモジュールを削除して再読み込み** します。

```python
import sys
for k in list(sys.modules):
    if k.startswith("unsloth"):
        del sys.modules[k]

from unsloth import FastLanguageModel
```

### 補足
`importlib_version` は実際には存在していたので、  
「読み込みの途中状態（キャッシュ）」が原因の可能性が高いです。

---

## 3) `from unsloth import FastLanguageModel` が長時間止まる

### 症状
ログが流れ続けて、数分以上待っても戻らない。

### まず確認すること
GPU が使えるかを確認します。

```python
import torch
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
```

### 今回の結果
GPU は認識されていました（`True` / `1`）。

### 補足
`No module named 'triton_kernels'` は **速度が少し落ちるだけ** で致命的ではありません。  
必要なら後で `triton-kernels` を入れると改善します。

---

## 4) Jupyter が別の Python を使ってしまう

### どう直すか
カーネルを登録して選び直します。

```bash
./.maLM25main/bin/python -m ipykernel install --user --name maLM25main --display-name "maLM25main (uv)"
```

Jupyter の `Kernel -> Change Kernel` で `maLM25main (uv)` を選択。
