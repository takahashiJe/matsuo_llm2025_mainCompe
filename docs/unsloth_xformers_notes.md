# Unsloth と xFormers の内部挙動まとめ（初心者向け）

このメモは、今回のエラー原因を後で振り返れるように、前提知識から整理したものです。

## 1. 前提: 学習で一番重い処理は Attention
大規模言語モデルでは、各トークンが他のトークンに「どれだけ注目するか」を計算する
**Attention** が最も重く、メモリも使います。

この Attention を **どの実装で計算するか**によって、
学習の安定性・速度・VRAM使用量が大きく変わります。

## 2. Attention の代表的な実装
代表的な実装は次の3つです。

1. **SDPA (PyTorch標準)**
   - 安定だがメモリ使用量が大きいことがある

2. **xFormers memory_efficient_attention**
   - 省メモリで高速
   - ただし**入力形状やGPUによって未対応カーネルがある**

3. **FlashAttention**
   - さらに高速で省メモリ
   - インストールや互換性の条件が厳しい

## 3. xFormers の特徴と注意点
xFormers は「メモリ効率の良い Attention 実装」です。
ただし注意点があります。

- **入力形状**や**GQA構造**によって対応できるカーネルが変わる
- 対応外だと「forward は通るが backward で落ちる」ことがある

今回の
`No operator found for memory_efficient_attention_backward`
は、**xFormers が対応カーネルを見つけられなかった**ことが原因でした。

## 4. Unsloth の役割
Unsloth は、Hugging Face / TRL の学習を高速化するライブラリです。
内部で次のようなことを行います。

- Attention を xFormers / FlashAttention に切り替える
- QKV/O/MLP などの層を高速パッチに差し替える
- 学習時と推論時で異なる最適化ルートを使う

つまり **中身をかなり書き換える高速化ライブラリ** です。

## 5. Unsloth の高速パッチが効かないケース
Unsloth は以下の条件で高速パッチを無効化します。

- **LoRA dropout が 0 以外**

ログに
```
Dropout = 0 is supported for fast patching. You are using dropout = 0.05.
```
と出る場合、**高速パッチが効いていません**。

この状態ではメモリ使用量が増え、OOM が起きやすくなります。

## 6. Phase1 -> Phase2 の LoRA 再読み込み問題
Phase1 と Phase2 では、内部の挙動が変わります。

- Phase1: **LoRA を新規作成**
- Phase2: **既存 LoRA を読み込み直して再学習**

この差で Unsloth の
`for_training / for_inference` 切替が噛み合わないことがあり、
**loss が grad を持たず backward で落ちる**状況が発生しました。

## 7. SDPA に切り替えると OOM しやすい
xFormers を無効化すると Attention は SDPA になります。
SDPA は安定ですが、**VRAM使用量が増える**ため、
同じ max_length でも OOM が起きやすくなります。

## 8. 今回のエラーの因果関係まとめ
今回の問題は主に次の2系統です。

1. **grad が消える問題**
   - Phase2 の LoRA 再読み込みで Unsloth の内部状態がズレる
   - loss が grad を持たず `RuntimeError` になる

2. **OOM 問題**
   - xFormers 無効化 -> SDPA 使用
   - SDPA の高メモリ消費で OOM

## 9. 今後の安定運用のポイント

- xFormers を有効にする
- Unsloth の高速パッチが効く条件を満たす
- Phase1->Phase2 を **単一runで連続実行**し、LoRA再読み込みを避ける
- OOM 対策として `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を使う

---

必要なら、このメモを README や docs に統合して運用資料にできます。
