# kabu-watch

日本株のテクニカル分析ダッシュボード（GitHub Actions + GitHub Pages で自動更新）。

- **`docs/index.html`** — 保有銘柄・マーケット指数・スクリーナー候補・急騰銘柄を一目均衡表 / RSI / MACD で分析するダッシュボード。平日場中10分おき＋大引け後に自動更新。
- **`docs/scan.html`** — 📷 **スクショで売り時診断**。SBI証券などの保有証券画面をスクショして読み込むと、写っている銘柄コードをブラウザ内OCR（Tesseract.js）で認識し、数量・取得単価を確認するだけで一目均衡表 / RSI / MACD ベースの売り時の目安を診断します。画像は外部送信せず端末内のみで処理。ホーム画面に追加してアプリのように使えます（PWA対応）。

## 使い方

1. GitHub Pages で公開されているダッシュボード（`docs/index.html`）をスマホで開く
2. ホーム画面に追加すると、次回からアプリのように起動できます
3. ダッシュボード右上の「📷 スクショで売り時診断」からスクショ診断ツールへ

## 開発

```
pip install -r requirements.txt
python screener.py          # docs/index.html を生成
python scripts/make_icons.py docs   # PWAアイコンを再生成（Pillow不要）
```

⚠ 本ツールはテクニカル指標に基づく機械的な目安を表示するもので、投資助言ではありません。最終的な売買判断はご自身の責任で行ってください。
