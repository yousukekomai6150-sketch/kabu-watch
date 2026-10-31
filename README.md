# kabu-watch

日本株のテクニカル分析ダッシュボード（GitHub Actions + GitHub Pages で自動更新）。

- **`docs/index.html`** — 保有銘柄・マーケット指数・スクリーナー候補・急騰銘柄を一目均衡表 / RSI / MACD で分析するダッシュボード。平日場中10分おき＋大引け後に自動更新。
- **`docs/scan.html`** — 📷 **スクショで売り時診断**。SBI証券などの保有証券画面をスクショして読み込むと、写っている銘柄コードをブラウザ内OCR（Tesseract.js）で認識し、数量・取得単価を確認するだけで一目均衡表 / RSI / MACD ベースの売り時の目安を診断します。画像は外部送信せず端末内のみで処理。ホーム画面に追加してアプリのように使えます（PWA対応）。

- **`docs/kronos.html`** — 🔮 **Kronos 検証ラボ**。ローソク足AI「[Kronos](https://github.com/shiyu-coder/Kronos)」（清華大学・AAAI 2026）が本当に当たるかを確かめる実験。平日17:20 JSTに監視銘柄の5営業日先までを予測して記録し、後日の実績と答え合わせします。比較相手は「流れ継続」「全部上がる」という単純な予想で、20営業日分たまると判定を表示。予測は売買判断に使いません。

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

### Kronos 検証ラボ

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-kronos.txt
git clone https://github.com/shiyu-coder/Kronos kronos_src
git -C kronos_src checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
PYTHONPATH=kronos_src python kronos_lab.py   # docs/kronos.html と docs/data/kronos_log.json を更新
```

設定は `config.json` の `kronos`。GitHub Actions では `.github/workflows/kronos.yml` が平日17:20 JSTに実行（Actions タブから手動実行も可）。金曜は LINE に週次レポートを送ります。

⚠ 本ツールはテクニカル指標に基づく機械的な目安を表示するもので、投資助言ではありません。最終的な売買判断はご自身の責任で行ってください。
