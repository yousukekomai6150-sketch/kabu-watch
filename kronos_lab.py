#!/usr/bin/env python3
"""Kronos 検証ラボ — ローソク足AI「Kronos」の予測を毎日記録し、実績と答え合わせする。

目的は「本当に当たるのか」を自分の監視銘柄で確かめること。予測は売買判断に使わない。

平日 17:20 JST に GitHub Actions（.github/workflows/kronos.yml）で実行:
  1. 過去の予測に、その後の実際の値動きを書き込む（答え合わせ）
  2. 監視銘柄の日足から Kronos-small で数営業日先までの終値を予測して記録
  3. 単純な予想（ベースライン）と比べた成績を docs/kronos.html に書き出す

ベースライン:
  - 流れ継続 … 直近の値動きの向きがそのまま続くと予想（翌日なら前日比、5日後なら過去5日の騰落）
  - 全部上がる … すべての銘柄が上がると予想
"""

from __future__ import annotations

import html
import json
import math
import os
import sys
import time
import zlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

JST = pytz.timezone("Asia/Tokyo")
CONFIG_PATH = "config.json"
LOG_PATH = "docs/data/kronos_log.json"
PAGE_PATH = "docs/kronos.html"
PAGES_URL = "https://yousukekomai6150-sketch.github.io/kabu-watch/kronos.html"
MARKET_CLOSE = (15, 45)  # 東証の大引け15:30＋余裕。これより前に取れた当日足は未確定なので使わない

DEFAULTS = {
    "model": "NeoQuasar/Kronos-small",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "model_revision": None,      # Hugging Face のコミットを固定したいときに指定
    "tokenizer_revision": None,
    "universe": ["holdings", "candidates", "tick_candidates", "scan_universe"],
    "lookback": 400,             # 入力する日足の本数（Kronos-small の上限は512）
    "horizon": 5,                # 何営業日先まで予測するか
    "sample_count": 5,           # 予測を何回サンプリングして平均するか
    "temperature": 1.0,
    "top_p": 0.9,
    "min_days_for_verdict": 20,  # 判定を出すのに必要な答え合わせ済みの営業日数
    "top_n": 3,                  # 「予測上位N銘柄」の超過リターンを測る N
    "line_weekly": True,         # 金曜の実行時に LINE で週次レポートを送る
}


# ── 設定・ログ ────────────────────────────────────────────────────────────────

def load_config(path: str = CONFIG_PATH) -> tuple[dict, list[tuple[str, str]]]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    kcfg = {**DEFAULTS, **cfg.get("kronos", {})}
    universe: list[tuple[str, str]] = []
    seen: set = set()
    for key in kcfg["universe"]:
        for item in cfg.get(key, []):
            sym = item["symbol"]
            if sym.startswith("^") or sym in seen:  # 指数は銘柄間比較に混ぜない
                continue
            seen.add(sym)
            universe.append((sym, item.get("name", sym)))
    return kcfg, universe


def load_log(path: str = LOG_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {"runs": []}, "records": []}


def save_log(log: dict, path: str = LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, separators=(",", ":"))


# ── データ ────────────────────────────────────────────────────────────────────

def fetch_daily(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    return yf.Ticker(symbol).history(period="3y", interval="1d", auto_adjust=True)


def normalize_ohlcv(df: pd.DataFrame | None, now_jst: datetime) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns=str.lower)
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols)
    df = df[cols].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(JST).tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[df["close"] > 0].astype(float)
    today = pd.Timestamp(now_jst.date())
    if len(df) and df.index[-1] == today and (now_jst.hour, now_jst.minute) < MARKET_CLOSE:
        df = df.iloc[:-1]  # 場中に取れた当日足は未確定
    return df


def next_trading_days(last: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """last の翌営業日から n 日分（土日・祝日・年末年始を除く）。Kronos の時刻特徴量に使う。"""
    try:
        import jpholiday
        is_holiday = jpholiday.is_holiday
    except ImportError:
        is_holiday = lambda d: False  # noqa: E731
    out: list[pd.Timestamp] = []
    d = pd.Timestamp(last)
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() >= 5 or is_holiday(d.date()):
            continue
        if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3):
            continue
        out.append(d)
    return out


def pct(a: float, b: float) -> float:
    return round((a / b - 1) * 100, 4)


# ── Kronos ────────────────────────────────────────────────────────────────────

def load_predictor(kcfg: dict):
    """Kronos-small を読み込む。PYTHONPATH に Kronos のソース（固定コミット）が必要。"""
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tokenizer = KronosTokenizer.from_pretrained(kcfg["tokenizer"], revision=kcfg.get("tokenizer_revision"))
    model = Kronos.from_pretrained(kcfg["model"], revision=kcfg.get("model_revision"))
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def resolve_model_sha(kcfg: dict) -> str | None:
    """実際に使ったモデルの Hugging Face コミットを記録（再現性のため）。失敗しても続行。"""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(kcfg["model"], revision=kcfg.get("model_revision"))
        return info.sha
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] model sha: {e}", file=sys.stderr)
        return None


def _seed(symbol: str, date_str: str) -> None:
    """同じ銘柄・同じ日なら同じ予測になるよう乱数を固定。"""
    s = zlib.crc32(f"{symbol}|{date_str}".encode())
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


def predict_closes(predictor, df: pd.DataFrame, kcfg: dict) -> list[float]:
    x = df.tail(kcfg["lookback"])
    x_ts = pd.Series(x.index)
    y_ts = pd.Series(next_trading_days(x.index[-1], kcfg["horizon"]))
    out = predictor.predict(
        df=x.reset_index(drop=True),
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=kcfg["horizon"],
        T=kcfg["temperature"],
        top_p=kcfg["top_p"],
        sample_count=kcfg["sample_count"],
        verbose=False,
    )
    return [float(v) for v in out["close"].tolist()]


# ── 答え合わせ ────────────────────────────────────────────────────────────────

def fill_actuals(records: list[dict], data: dict[str, pd.DataFrame]) -> int:
    """予測日より後の実際の終値から、翌日・h日後の騰落率を書き込む。"""
    filled = 0
    for r in records:
        if r.get("act_hd") is not None:
            continue
        df = data.get(r["symbol"])
        if df is None or df.empty:
            continue
        base = pd.Timestamp(r["date"])
        after = df[df.index > base]
        if after.empty:
            continue
        # 配当・分割の調整で過去の株価が書き換わることがあるので、同じ系列の基準日終値を使う
        base_close = float(df["close"].get(base, r["close"]))
        if r.get("act_1d") is None:
            r["act_1d"] = pct(after["close"].iloc[0], base_close)
            filled += 1
        h = r.get("h", 5)
        if len(after) >= h:
            r["act_hd"] = pct(after["close"].iloc[h - 1], base_close)
            filled += 1
    return filled


def mean_ci(values, inflate: float = 1.0) -> dict:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(np.mean(v))
    if n < 2:
        return {"mean": m, "lo": None, "hi": None, "n": n}
    se = float(np.std(v, ddof=1)) / math.sqrt(n) * inflate
    return {"mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": n}


def score(records: list[dict], horizon_key: str, h: int, top_n: int) -> dict:
    """horizon_key: '1d'（翌日）または 'hd'（h営業日後）。"""
    pk, ak, mk = f"pred_{horizon_key}", f"act_{horizon_key}", f"past_{horizon_key}"
    rows = [r for r in records if r.get(ak) is not None and r.get(pk) is not None and r.get(mk) is not None]
    if not rows:
        return {"n": 0, "days": 0}
    df = pd.DataFrame(rows)
    up = df[ak] > 0
    df["hit_k"] = (df[pk] > 0) == up
    df["hit_m"] = (df[mk] > 0) == up
    df["hit_u"] = up
    daily = df.groupby("date").agg(hit_k=("hit_k", "mean"), hit_m=("hit_m", "mean"),
                                   hit_u=("hit_u", "mean"), n=("hit_k", "size"))
    ics, spreads, ic_by_day = [], [], {}
    for d, g in df.groupby("date"):
        if len(g) >= 5 and g[pk].nunique() > 1 and g[ak].nunique() > 1:
            ic = float(g[pk].corr(g[ak], method="spearman"))
            ics.append(ic)
            ic_by_day[d] = ic
        if len(g) >= top_n * 2:
            spreads.append(float(g.nlargest(top_n, pk)[ak].mean() - g[ak].mean()))
    # 複数日先の予測は評価期間が重なり、日々の結果が独立でなくなる。誤差幅を √h 倍に広げて慎重に判定する
    inflate = math.sqrt(h) if horizon_key == "hd" else 1.0
    recent = []
    for d, row in daily.tail(10).iloc[::-1].iterrows():
        recent.append({"date": d, "hit_k": row.hit_k, "hit_m": row.hit_m, "n": int(row.n), "ic": ic_by_day.get(d)})
    return {
        "n": int(len(df)),
        "days": int(len(daily)),
        "hit": {"kronos": float(df.hit_k.mean()), "momentum": float(df.hit_m.mean()), "up": float(df.hit_u.mean())},
        "diff_m": mean_ci((daily.hit_k - daily.hit_m).tolist(), inflate),
        "diff_u": mean_ci((daily.hit_k - daily.hit_u).tolist(), inflate),
        "ic": mean_ci(ics, inflate),
        "spread": mean_ci(spreads, inflate),
        "recent": recent,
    }


def verdict(s: dict, min_days: int) -> tuple[str, str]:
    days = s.get("days", 0)
    if days < min_days:
        return "collecting", f"データ収集中（判定まであと{min_days - days}営業日）"
    best = "momentum" if s["hit"]["momentum"] >= s["hit"]["up"] else "up"
    diff = s["diff_m"] if best == "momentum" else s["diff_u"]
    ic = s["ic"]
    if diff["lo"] is not None and diff["lo"] > 0 and ic["lo"] is not None and ic["lo"] > 0:
        return "good", "有望：単純な予想より統計的にはっきり当たっている"
    if diff["hi"] is not None and diff["hi"] < 0:
        return "bad", "単純な予想に負けている"
    return "neutral", "差なし：偶然と区別できない"


# ── ページ生成 ────────────────────────────────────────────────────────────────

VERDICT_COLOR = {"collecting": "#888", "good": "#ffd600", "neutral": "#7c83ff", "bad": "#ff7c7c"}


def _c(x: float | None) -> str:
    """SBI風: 上昇=赤、下落=緑。"""
    if x is None:
        return "#666"
    return "#ef5350" if x > 0 else ("#26a69a" if x < 0 else "#aaa")


def _p(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{'+' if x > 0 else ''}{x:.{digits}f}%"


def _rate(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _ci(ci: dict, scale: float = 1.0, unit: str = "", digits: int = 1) -> str:
    if ci.get("mean") is None:
        return "—"
    m = ci["mean"] * scale
    s = f"{'+' if m > 0 else ''}{m:.{digits}f}{unit}"
    if ci.get("lo") is not None:
        lo, hi = ci["lo"] * scale, ci["hi"] * scale
        s += f'<span class="ci"> (95%: {lo:+.{digits}f}〜{hi:+.{digits}f})</span>'
    return s


def _price(p: float) -> str:
    return f"{p:,.1f}" if p < 100 else f"{p:,.0f}"


def render_score_card(title: str, s: dict, v: tuple[str, str], top_n: int) -> str:
    color = VERDICT_COLOR[v[0]]
    if s.get("n", 0) == 0:
        body = '<div class="mute">まだ答え合わせできた予測がありません</div>'
    else:
        hit = s["hit"]
        body = f"""
<table class="t">
<tr><th>的中率（上がる/下がるの向き）</th><th></th></tr>
<tr><td>Kronos</td><td class="r b">{_rate(hit['kronos'])}</td></tr>
<tr><td>流れ継続（単純予想）</td><td class="r">{_rate(hit['momentum'])}</td></tr>
<tr><td>全部「上がる」</td><td class="r">{_rate(hit['up'])}</td></tr>
<tr><td>Kronos − 流れ継続</td><td class="r">{_ci(s['diff_m'], 100, 'pt')}</td></tr>
<tr><td>Kronos − 全部上がる</td><td class="r">{_ci(s['diff_u'], 100, 'pt')}</td></tr>
<tr><td>順位相関 RankIC</td><td class="r">{_ci(s['ic'], 1, '', 3)}</td></tr>
<tr><td>予測上位{top_n}銘柄の超過リターン</td><td class="r">{_ci(s['spread'], 1, '%', 2)}</td></tr>
</table>
<div class="mute">{s['days']}営業日・{s['n']}件で評価</div>"""
    return f"""<div class="card">
<div class="ct">{title}</div>
<div class="vd" style="color:{color};border-color:{color}">{html.escape(v[1])}</div>
{body}
</div>"""


def render_recent(s: dict) -> str:
    if not s.get("recent"):
        return ""
    rows = []
    for r in s["recent"]:
        day = r["date"][5:].replace("-", "/")
        ic = "—" if r["ic"] is None else format(r["ic"], "+.2f")
        rows.append(f'<tr><td>{day}</td><td class="r">{_rate(r["hit_k"])}</td>'
                    f'<td class="r">{_rate(r["hit_m"])}</td><td class="r">{ic}</td></tr>')
    return ('<div class="card"><div class="ct">直近の答え合わせ（翌日の向き）</div>'
            '<table class="t"><tr><th>予測日</th><th class="r">Kronos</th><th class="r">流れ継続</th>'
            f'<th class="r">IC</th></tr>{"".join(rows)}</table></div>')


def render_predictions(latest: list[dict], h: int) -> str:
    if not latest:
        return '<div class="card mute">まだ予測がありません（平日17:20 JSTに自動実行）</div>'
    latest = sorted(latest, key=lambda r: -r["pred_hd"])

    def row(r: dict) -> str:
        return (f'<tr><td>{html.escape(r["name"])}<div class="sym">{r["symbol"].replace(".T", "")}</div></td>'
                f'<td class="r">{_price(r["close"])}</td>'
                f'<td class="r" style="color:{_c(r["pred_1d"])}">{_p(r["pred_1d"])}</td>'
                f'<td class="r b" style="color:{_c(r["pred_hd"])}">{_p(r["pred_hd"])}</td></tr>')

    head = f'<tr><th>銘柄</th><th class="r">終値</th><th class="r">翌日</th><th class="r">{h}日後</th></tr>'
    top = "".join(row(r) for r in latest[:5])
    bottom = ""
    if len(latest) > 10:
        bottom = ('<div class="mute" style="margin:8px 0 6px">予測下位5</div>'
                  f'<table class="t pt">{head}{"".join(row(r) for r in latest[-5:])}</table>')
    all_rows = "".join(row(r) for r in latest)
    day = latest[0]["date"].replace("-", "/")
    return (f'<div class="card"><div class="ct">{day} 終値時点の予測（{len(latest)}銘柄）</div>'
            f'<div class="mute" style="margin-bottom:6px">予測上位5</div><table class="t pt">{head}{top}</table>'
            f'{bottom}<details><summary>全銘柄を見る</summary><table class="t pt">{head}{all_rows}</table></details>'
            '</div>')


def render_page(log: dict, kcfg: dict, now_jst: datetime, notes: list[str]) -> str:
    records = log["records"]
    h = kcfg["horizon"]
    s1 = score(records, "1d", h, kcfg["top_n"])
    sh = score(records, "hd", h, kcfg["top_n"])
    v1 = verdict(s1, kcfg["min_days_for_verdict"])
    vh = verdict(sh, kcfg["min_days_for_verdict"])
    latest_date = max((r["date"] for r in records), default=None)
    latest = [r for r in records if r["date"] == latest_date] if latest_date else []
    runs = log["meta"].get("runs", [])
    last_run = runs[-1] if runs else {}
    notes_html = "".join(f'<div class="warn">{html.escape(n)}</div>' for n in notes)
    model_sha = (last_run.get("model_sha") or "不明")[:12]
    kronos_commit = (last_run.get("kronos_commit") or "不明")[:12]

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kronos 検証ラボ | kabu-watch</title>
<meta name="theme-color" content="#080812">
<link rel="icon" href="icon-192.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080812;color:#e0e0f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  max-width:480px;margin:0 auto;padding:12px;font-size:13px;line-height:1.5}}
a{{color:#7c83ff;text-decoration:none}}
h2{{font-size:13px;letter-spacing:.08em;color:#666;margin:16px 0 8px}}
.card{{background:#10101e;border:1px solid #22223a;border-radius:8px;padding:10px 12px;margin-bottom:10px}}
.ct{{font-size:12px;color:#aab;font-weight:600;margin-bottom:6px}}
.vd{{border:1px solid;border-radius:6px;padding:6px 8px;font-weight:700;font-size:14px;margin-bottom:8px}}
.t{{width:100%;border-collapse:collapse;font-size:12px}}
.t th{{color:#667;font-weight:500;text-align:left;padding:3px 0;border-bottom:1px solid #22223a}}
.t td{{padding:4px 0;border-bottom:1px solid #16162a;vertical-align:top}}
.pt{{table-layout:fixed}} .pt th:first-child{{width:40%}}
.r{{text-align:right}} .b{{font-weight:700}}
.ci{{color:#667;font-size:10px;font-weight:400}}
.sym{{color:#556;font-size:10px}}
.mute{{color:#667;font-size:11px}}
.warn{{background:#1a0505;border:1px solid #cc2200;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:#ff5533}}
.info{{background:#0d0d24;border:1px solid #2a2a55;border-radius:4px;padding:6px 10px;margin-bottom:10px;font-size:11px;color:#99a}}
details summary{{cursor:pointer;color:#7c83ff;font-size:12px;margin-top:8px}}
details .t{{margin-top:6px}}
.foot li{{margin:2px 0 2px 16px}}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:flex-start;padding:4px 0 10px">
  <div>
    <div style="font-size:20px;font-weight:800;color:#7c83ff">🔮 Kronos 検証ラボ</div>
    <a href="index.html" style="font-size:11px">← ダッシュボードへ</a>
  </div>
  <div style="font-size:10px;color:#555;text-align:right">更新: {now_jst.strftime("%-m/%-d %H:%M")} JST</div>
</div>
<div class="info">ローソク足AI「Kronos」が本当に当たるかを確かめる実験です。<b>予測を売買判断に使わないでください。</b>
答え合わせが{kcfg["min_days_for_verdict"]}営業日分たまると判定が出ます（{h}日後の判定は予測から{h}営業日遅れ）。</div>
{notes_html}
<h2>成績</h2>
{render_score_card(f"{h}営業日後（メイン）", sh, vh, kcfg["top_n"])}
{render_score_card("翌営業日", s1, v1, kcfg["top_n"])}
{render_recent(s1)}
<h2>最新の予測</h2>
{render_predictions(latest, h)}
<h2>仕組み</h2>
<div class="card foot mute">
<ul>
<li>モデル: {html.escape(kcfg["model"])}（HF {model_sha}）／ソース: shiyu-coder/Kronos @{kronos_commit}</li>
<li>入力: 日足{kcfg["lookback"]}本（Yahoo Finance）。{kcfg["sample_count"]}回サンプリングして平均</li>
<li>的中率: 予測した向き（上/下）と実際の向きが一致した割合</li>
<li>RankIC: 銘柄間で「予測の順位」と「実際の順位」がどれだけ一致したか（0なら無関係）</li>
<li>判定: 「Kronos − 単純予想の良い方」と RankIC の95%区間がどちらも0より上なら「有望」</li>
<li>{h}日後の評価は期間が重なるため、誤差幅を√{h}倍に広げて慎重に判定</li>
<li>1か月程度では大きな差しか見分けられません。3か月たまると精度が上がります</li>
</ul>
</div>
<div style="text-align:center;font-size:10px;color:#333;padding:16px 0 8px">
  自動実行: 平日 17:20 JST | <a href="https://github.com/yousukekomai6150-sketch/kabu-watch" style="color:#444">GitHub</a>
</div>
</body>
</html>"""


# ── LINE ──────────────────────────────────────────────────────────────────────

def weekly_line_message(log: dict, kcfg: dict, now_jst: datetime) -> str:
    h = kcfg["horizon"]
    lines = [f"🔮 Kronos検証 週次レポート {now_jst.strftime('%-m/%-d')}"]
    for title, key in ((f"{h}日後", "hd"), ("翌日", "1d")):
        s = score(log["records"], key, h, kcfg["top_n"])
        v = verdict(s, kcfg["min_days_for_verdict"])
        lines += ["", f"【{title}】{v[1]}"]
        if s.get("n"):
            hit = s["hit"]
            lines.append(f"的中率 Kronos {_rate(hit['kronos'])} / 流れ継続 {_rate(hit['momentum'])} / 全部上がる {_rate(hit['up'])}")
    lines += ["", PAGES_URL]
    return "\n".join(lines)


# ── メイン ────────────────────────────────────────────────────────────────────

def main(now_jst: datetime | None = None, fetch=fetch_daily, predictor_factory=load_predictor,
         config_path: str = CONFIG_PATH, log_path: str = LOG_PATH, page_path: str = PAGE_PATH,
         send_line_fn=None) -> int:
    t0 = time.time()
    now_jst = now_jst or datetime.now(JST)
    kcfg, universe = load_config(config_path)
    log = load_log(log_path)
    records: list[dict] = log["records"]
    notes: list[str] = []

    # 1. データ取得（監視銘柄＋答え合わせ待ちの銘柄）
    names = dict(universe)
    pending = {r["symbol"] for r in records if r.get("act_hd") is None}
    data: dict[str, pd.DataFrame] = {}
    for sym in list(names) + sorted(pending - set(names)):
        try:
            data[sym] = normalize_ohlcv(fetch(sym), now_jst)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] fetch {sym}: {e}", file=sys.stderr)

    # 2. 答え合わせ
    filled = fill_actuals(records, data)

    # 3. 予測
    done = {(r["date"], r["symbol"]) for r in records}
    todo = []
    for sym, name in universe:
        df = data.get(sym)
        if df is None or len(df) < 120:
            continue
        last = df.index[-1]
        if (pd.Timestamp(now_jst.date()) - last).days > 7:  # 売買停止・データ途絶
            continue
        if (last.strftime("%Y-%m-%d"), sym) in done:
            continue
        todo.append((sym, name, df))

    n_pred = 0
    model_sha = None
    if todo:
        try:
            predictor = predictor_factory(kcfg)
            if predictor_factory is load_predictor:
                model_sha = resolve_model_sha(kcfg)
        except Exception as e:  # noqa: BLE001
            predictor = None
            msg = f"Kronosの読み込みに失敗したため、今回は予測をスキップしました（{type(e).__name__}: {e}）"
            notes.append(msg)
            print(f"::error::{msg}")
        h = kcfg["horizon"]
        for sym, name, df in todo if predictor else []:
            date_str = df.index[-1].strftime("%Y-%m-%d")
            try:
                _seed(sym, date_str)
                closes = predict_closes(predictor, df, kcfg)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] predict {sym}: {e}", file=sys.stderr)
                continue
            c0 = float(df["close"].iloc[-1])
            if not all(math.isfinite(c) and c > 0 for c in closes):
                print(f"[WARN] predict {sym}: invalid output {closes}", file=sys.stderr)
                continue
            records.append({
                "date": date_str, "symbol": sym, "name": name, "close": round(c0, 2), "h": h,
                "pred": [round(c, 2) for c in closes],
                "pred_1d": pct(closes[0], c0), "pred_hd": pct(closes[h - 1], c0),
                "past_1d": pct(c0, float(df["close"].iloc[-2])),
                "past_hd": pct(c0, float(df["close"].iloc[-1 - h])),
                "act_1d": None, "act_hd": None,
            })
            n_pred += 1

    secs = round(time.time() - t0, 1)
    if todo or filled:
        runs = log["meta"].setdefault("runs", [])
        prev_sha = runs[-1].get("model_sha") if runs else None
        runs.append({
            "at": now_jst.strftime("%Y-%m-%d %H:%M"), "predicted": n_pred, "filled": filled, "secs": secs,
            "model": kcfg["model"], "model_sha": model_sha or prev_sha,
            "kronos_commit": os.environ.get("KRONOS_COMMIT"),
        })
        log["meta"]["runs"] = runs[-400:]
    save_log(log, log_path)

    page = render_page(log, kcfg, now_jst, notes)
    os.makedirs(os.path.dirname(page_path) or ".", exist_ok=True)
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[OK] predicted={n_pred} filled={filled} records={len(records)} secs={secs}")

    # 4. LINE（金曜のみ・週次）
    token, user_id = os.environ.get("LINE_CHANNEL_TOKEN", ""), os.environ.get("LINE_USER_ID", "")
    if kcfg["line_weekly"] and now_jst.weekday() == 4 and token and user_id:
        if send_line_fn is None:
            from screener import send_line as send_line_fn
        send_line_fn(token, user_id, weekly_line_message(log, kcfg, now_jst))
        print("[OK] LINE weekly report sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
