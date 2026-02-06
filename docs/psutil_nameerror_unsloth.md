# `NameError: name 'psutil' is not defined`（Unsloth）解説

このページは、`train_lora_sanity.py` 実行時に出た以下エラーを理解するためのメモです。

```text
NameError: name 'psutil' is not defined
```

## 1) エラーの意味

Python の基本ルールとして、変数やモジュールは「定義/import された後」しか使えません。  
今回の意味はシンプルで、**`psutil` を使っているのに import されていない**、です。

## 2) どこで起きていたか（今回の実例）

スタックトレース上は、あなたの業務ロジックではなく **Unsloth が生成したキャッシュコード** で落ちていました。

- `unsloth_compiled_cache/UnslothSFTTrainer.py`
- `_prepare_dataset` 内で `psutil.cpu_count()` を参照

つまり、
- `scripts/train_lora_sanity.py` → `scripts/train_lora.py` → `SFTTrainer`
- の内部で Unsloth 側コードに入り、そこで `psutil` 未定義例外が発生

という流れです。

## 3) なぜ `rm -rf unsloth_compiled_cache/` で直らないのか

`unsloth_compiled_cache/` を消すと、次回実行で Unsloth が同種コードを再生成します。  
再生成後も同じ条件に入ると、**同じ `psutil` 参照箇所**で再び落ちます。

要点:
- cache は「壊れたファイル」を消すには有効
- ただし今回のように「再生成ロジック側の分岐条件」が同じなら再発する

## 4) 今回の有効だった考え方

Unsloth 生成コードは、`dataset_num_proc` が未設定（`None`）だと自動判定のため `psutil` を使います。  
そのため、**`dataset_num_proc` を明示設定して `psutil` 分岐に入らない**ようにします。

実装上のポイント:
- `TrainingArguments` ではなく `trl.SFTConfig` を使って `dataset_num_proc` を正式フィールドで渡す
- 例: `dataset_num_proc=2`

## 5) 学び（再発防止チェックリスト）

1. `NameError` はまず「その名前がどこで定義されるべきか」を確認する。  
2. スタックトレースの**最下段の自作コードだけでなく、site-packages/生成コードも見る**。  
3. `rm -rf cache` は万能ではない。  
4. ライブラリ内部分岐を避けるため、設定値（今回なら `dataset_num_proc`）を明示する。  
5. `TrainingArguments` と `SFTConfig` のように、**型変換で引数が落ちる経路**を疑う。

## 6) 参考: この後に出た別エラーとの違い

`psutil` エラー解消後に出た `KeyError: 'text'` は別問題です。  
これは「データコラレータが `text` を期待しているが、前処理で `input_ids` 化されて `text` が無い」不整合です。

つまり:
- `NameError: psutil` = 依存/分岐設定の問題
- `KeyError: text` = データ前処理とコラレータ設計の不整合
