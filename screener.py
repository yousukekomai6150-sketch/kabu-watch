#!/usr/bin/env python3
"""Ichimoku screener for Japanese stocks — generates docs/index.html."""

import json
import os
import sys
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import requests

JST = pytz.timezone("Asia/Tokyo")


# ── Ichimoku ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period="6mo", interval="1d", auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:
        print(f"[WARN] {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c, o = df["High"], df["Low"], df["Close"], df["Open"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c,
         "tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b},
        index=df.index,
    )


def calc_signals(ich: pd.DataFrame) -> tuple:
    empty = {"雲抜け": False, "三役好転": False, "上昇トレンド": False, "転換GC": False}
    if len(ich) < 55:
        return 0, empty

    lat = ich.iloc[-1]
    sa, sb = lat["span_a"], lat["span_b"]
    if pd.isna(sa) or pd.isna(sb):
        return 0, empty

    cloud_top = max(sa, sb)
    kumo = bool(lat["close"] > cloud_top)

    chikou_ok = len(ich) > 26 and bool(lat["close"] > ich["close"].iloc[-27])
    tenkan_ok = (not pd.isna(lat["tenkan"])) and (not pd.isna(lat["kijun"])) and lat["tenkan"] > lat["kijun"]
    saneki    = bool(kumo and tenkan_ok and chikou_ok)
    josho     = bool((not pd.isna(lat["kijun"])) and lat["close"] > lat["kijun"])

    gc = False
    for i in range(1, min(4, len(ich))):
        cur, prv = ich.iloc[-i], ich.iloc[-i - 1]
        if all(not pd.isna(v) for v in [cur["tenkan"], cur["kijun"], prv["tenkan"], prv["kijun"]]):
            if prv["tenkan"] <= prv["kijun"] and cur["tenkan"] > cur["kijun"]:
                gc = True
                break

    sigs = {"雲抜け": kumo, "三役好転": saneki, "上昇トレンド": josho, "転換GC": gc}
    return sum(sigs.values()), sigs


# ── SVG Chart ─────────────────────────────────────────────────────────────────

def make_chart(ich: pd.DataFrame, w: int = 360, h: int = 180) -> str:
    n = min(60, len(ich))
    d = ich.iloc[-n:].reset_index(drop=True)
    NA = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">'
          f'<rect width="{w}" height="{h}" fill="#0d0d1a" rx="4"/>'
          f'<text x="50%" y="52%" fill="#555" text-anchor="middle" font-size="11">データなし</text></svg>')
    if len(d) < 3:
        return NA

    PT, PB, PL, PR = 16, 4, 4, 4
    cw, ch_ = w - PL - PR, h - PT - PB

    vals = []
    for col in ["low", "high", "span_a", "span_b"]:
        vals += d[col].dropna().tolist()
    if not vals:
        return NA

    pmin = min(vals) * 0.999
    pmax = max(vals) * 1.001
    prng = pmax - pmin or 1

    def py(p: float) -> float:
        return PT + ch_ * (1.0 - (p - pmin) / prng)

    slot = cw / n
    bw   = max(1.5, slot * 0.65)

    def bx(i: int) -> float:
        return PL + (i + 0.5) * slot

    parts = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">',
        f'<rect width="{w}" height="{h}" fill="#0d0d1a" rx="4"/>',
    ]

    # Price / pct label
    c0  = d["close"].iloc[-1]
    c1  = d["close"].iloc[-2] if len(d) > 1 else c0
    pct = (c0 / c1 - 1) * 100 if c1 else 0
    pc  = "#26a69a" if pct >= 0 else "#ef5350"
    ps  = "+" if pct >= 0 else ""
    lbl = f"{c0:,.0f}" if c0 >= 100 else f"{c0:.2f}"
    parts.append(f'<text x="{PL}" y="11" fill="#aaa" font-size="9" font-family="monospace">{lbl}</text>')
    parts.append(f'<text x="{w - PR}" y="11" fill="{pc}" font-size="9" font-family="monospace" text-anchor="end">{ps}{pct:.2f}%</text>')

    # Cloud polygon
    cloud_pts = [
        (i, d["span_a"].iloc[i], d["span_b"].iloc[i])
        for i in range(n)
        if not (pd.isna(d["span_a"].iloc[i]) or pd.isna(d["span_b"].iloc[i]))
    ]
    if len(cloud_pts) >= 2:
        top = " ".join(f"{bx(i):.1f},{py(max(sa, sb)):.1f}" for i, sa, sb in cloud_pts)
        bot = " ".join(f"{bx(i):.1f},{py(min(sa, sb)):.1f}" for i, sa, sb in reversed(cloud_pts))
        bull = sum(1 for _, sa, sb in cloud_pts if sa >= sb)
        fill = "#1a3d2a" if bull >= len(cloud_pts) / 2 else "#3d1a1a"
        parts.append(f'<polygon points="{top} {bot}" fill="{fill}" opacity="0.75"/>')

    # Span A / Span B lines
    def polyline(col: str, stroke: str, sw: float = 1.0) -> None:
        pts = [(bx(i), py(d[col].iloc[i])) for i in range(n) if not pd.isna(d[col].iloc[i])]
        if len(pts) > 1:
            ps_ = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{ps_}" fill="none" stroke="{stroke}" stroke-width="{sw}" opacity="0.8"/>')

    polyline("span_a", "#26a69a", 0.8)
    polyline("span_b", "#ef5350", 0.8)
    polyline("tenkan", "#4499ff", 1.2)
    polyline("kijun",  "#ff9944", 1.2)

    # Candlesticks
    for i in range(n):
        row = d.iloc[i]
        o_, c_, hh, ll = row["open"], row["close"], row["high"], row["low"]
        if any(pd.isna(v) for v in [o_, c_, hh, ll]):
            continue
        x_  = bx(i)
        col = "#26a69a" if c_ >= o_ else "#ef5350"
        parts.append(f'<line x1="{x_:.1f}" y1="{py(hh):.1f}" x2="{x_:.1f}" y2="{py(ll):.1f}" stroke="{col}" stroke-width="0.8"/>')
        yt  = py(max(o_, c_))
        yb  = py(min(o_, c_))
        bh  = max(1.0, yb - yt)
        parts.append(f'<rect x="{x_ - bw/2:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}"/>')

    parts.append("</svg>")
    return "\n".join(parts)


# ── HTML helpers ──────────────────────────────────────────────────────────────

SIGNAL_META = {
    "雲抜け":    {"color": "#2979ff", "bg": "#0d2a66"},
    "三役好転":  {"color": "#ffd600", "bg": "#4a3a00"},
    "上昇トレンド": {"color": "#00e676", "bg": "#003322"},
    "転換GC":    {"color": "#ea80fc", "bg": "#330044"},
}


def signal_badges(sigs: dict) -> str:
    badges = []
    for name, active in sigs.items():
        if active:
            m = SIGNAL_META[name]
            badges.append(
                f'<span style="background:{m["bg"]};color:{m["color"]};border:1px solid {m["color"]};'
                f'border-radius:4px;padding:2px 6px;font-size:11px;white-space:nowrap">{name}</span>'
            )
    return " ".join(badges) if badges else '<span style="color:#555;font-size:11px">シグナルなし</span>'


def fmt_price(p: float) -> str:
    return f"{p:,.0f}" if p >= 100 else f"{p:.2f}"


def card(content: str) -> str:
    return f'<div style="background:#111122;border-radius:8px;padding:12px;margin-bottom:12px">{content}</div>'


# ── LINE ──────────────────────────────────────────────────────────────────────

def send_line(token: str, user_id: str, message: str) -> None:
    try:
        r = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": [{"type": "text", "text": message}]},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[WARN] LINE API: {r.status_code} {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] LINE send failed: {e}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    now_jst = datetime.now(JST)
    os.makedirs("docs", exist_ok=True)

    # ── Fetch indices ─────────────────────────────────────────────────────────
    idx_data: list[dict] = []
    for idx in cfg["indices"]:
        df   = fetch_ohlcv(idx["symbol"])
        ich  = ichimoku(df) if len(df) >= 55 else pd.DataFrame()
        last = df["Close"].iloc[-1] if len(df) else None
        prev = df["Close"].iloc[-2] if len(df) > 1 else last
        pct  = ((last / prev - 1) * 100) if (last and prev) else None
        idx_data.append({
            "symbol": idx["symbol"],
            "name": idx["name"],
            "last": last,
            "pct": pct,
            "chart": make_chart(ich) if len(ich) >= 3 else "",
        })

    # ── Fetch holdings ────────────────────────────────────────────────────────
    hold_data: list[dict] = []
    for h in cfg["holdings"]:
        df   = fetch_ohlcv(h["symbol"])
        ich  = ichimoku(df) if len(df) >= 55 else pd.DataFrame()
        last = df["Close"].iloc[-1] if len(df) else None
        pnl  = ((last / h["cost"] - 1) * 100) if last else None
        _, sigs = calc_signals(ich) if len(ich) >= 55 else (0, {})
        hold_data.append({
            "symbol": h["symbol"],
            "name": h["name"],
            "cost": h["cost"],
            "last": last,
            "pnl": pnl,
            "stop_loss": h.get("stop_loss"),
            "sigs": sigs,
            "chart": make_chart(ich) if len(ich) >= 3 else "",
        })

    # ── Screen candidates ─────────────────────────────────────────────────────
    max_n = cfg.get("max_candidates", 10)
    results: list[dict] = []
    for cand in cfg["candidates"]:
        df  = fetch_ohlcv(cand["symbol"])
        if len(df) < 55:
            continue
        ich = ichimoku(df)
        score, sigs = calc_signals(ich)
        last = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df) > 1 else last
        pct  = (last / prev - 1) * 100 if prev else 0
        results.append({
            "symbol": cand["symbol"],
            "name": cand["name"],
            "last": last,
            "pct": pct,
            "score": score,
            "sigs": sigs,
            "chart": make_chart(ich),
        })

    results.sort(key=lambda x: (-x["score"], -x["pct"]))
    top = results[:max_n]

    # ── Build HTML ────────────────────────────────────────────────────────────
    def idx_html() -> str:
        items = []
        for d in idx_data:
            if d["last"] is None:
                items.append(card(f'<div style="color:#888">{d["name"]} — データ取得失敗</div>'))
                continue
            pc  = "#26a69a" if (d["pct"] or 0) >= 0 else "#ef5350"
            ps  = "+" if (d["pct"] or 0) >= 0 else ""
            pct_str = f'{ps}{d["pct"]:.2f}%' if d["pct"] is not None else "―"
            items.append(card(
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">'
                f'<span style="font-weight:600;font-size:14px">{d["name"]}</span>'
                f'<span style="font-size:11px;color:#666">{d["symbol"]}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">'
                f'<span style="font-size:20px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</span>'
                f'<span style="font-size:14px;color:{pc}">{pct_str}</span></div>'
                + (d["chart"] if d["chart"] else "")
            ))
        return "\n".join(items)

    def hold_html() -> str:
        items = []
        for d in hold_data:
            if d["last"] is None:
                items.append(card(f'<div style="color:#888">{d["name"]} — データ取得失敗</div>'))
                continue
            pc      = "#26a69a" if (d["pnl"] or 0) >= 0 else "#ef5350"
            ps      = "+" if (d["pnl"] or 0) >= 0 else ""
            pnl_str = f'{ps}{d["pnl"]:.2f}%' if d["pnl"] is not None else "―"
            warn    = ""
            if d["stop_loss"] is None:
                warn = ('<div style="background:#3a1a00;border:1px solid #ff6600;border-radius:4px;'
                        'padding:4px 8px;margin-top:6px;font-size:11px;color:#ff9944">⚠ 逆指値未設定</div>')
            items.append(card(
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
                f'<span style="font-weight:600;font-size:14px">{d["name"]}</span>'
                f'<span style="font-size:11px;color:#666">{d["symbol"]}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
                f'<span style="font-size:18px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</span>'
                f'<span style="font-size:14px;color:{pc}">{pnl_str}</span></div>'
                f'<div style="font-size:11px;color:#666;margin-bottom:6px">'
                f'取得 {fmt_price(d["cost"])} 円'
                + (f' | 逆指値 {fmt_price(d["stop_loss"])} 円' if d["stop_loss"] else "")
                + f'</div>'
                + (f'<div style="margin-bottom:6px">{signal_badges(d["sigs"])}</div>' if d["sigs"] else "")
                + (d["chart"] if d["chart"] else "")
                + warn
            ))
        return "\n".join(items)

    def screen_html() -> str:
        if not top:
            return card('<div style="color:#888;text-align:center">候補銘柄なし</div>')
        items = []
        for rank, d in enumerate(top, 1):
            pc  = "#26a69a" if d["pct"] >= 0 else "#ef5350"
            ps  = "+" if d["pct"] >= 0 else ""
            score_color = ["#333", "#1a3d2a", "#1a4a2a", "#1a5a2a", "#00cc66"][min(d["score"], 4)]
            items.append(card(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<span style="background:{score_color};color:#fff;border-radius:4px;padding:2px 7px;'
                f'font-size:13px;font-weight:700;min-width:22px;text-align:center">{d["score"]}</span>'
                f'<div><div style="font-weight:600;font-size:14px">{d["name"]}</div>'
                f'<div style="font-size:11px;color:#666">{d["symbol"]}</div></div></div>'
                f'<div style="text-align:right">'
                f'<div style="font-size:16px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</div>'
                f'<div style="font-size:12px;color:{pc}">{ps}{d["pct"]:.2f}%</div></div></div>'
                f'<div style="margin-bottom:8px">{signal_badges(d["sigs"])}</div>'
                + (d["chart"] if d["chart"] else "")
            ))
        return "\n".join(items)

    updated_str = now_jst.strftime("%Y年%m月%d日 %H:%M JST")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kabu-watch</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080812;color:#e0e0f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  max-width:480px;margin:0 auto;padding:12px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#666;margin:16px 0 8px}}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0 12px">
  <div style="font-size:20px;font-weight:800;color:#7c83ff">kabu-watch</div>
  <div style="font-size:11px;color:#555">{updated_str}</div>
</div>

<h2>マーケット</h2>
{idx_html()}

<h2>保有銘柄</h2>
{hold_html()}

<h2>スクリーナー TOP{len(top)}</h2>
<div style="font-size:11px;color:#555;margin-bottom:8px">スコア = 雲抜け + 三役好転 + 上昇トレンド + 転換GC (最大4)</div>
{screen_html()}

<div style="text-align:center;font-size:10px;color:#333;padding:16px 0 8px">
  自動更新: 平日 16:30 JST | <a href="https://github.com/yousukekomai6150-sketch/kabu-watch" style="color:#444">GitHub</a>
</div>
</body>
</html>"""

    out_path = "docs/index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {out_path} written ({len(html):,} bytes)")

    # ── LINE notification ─────────────────────────────────────────────────────
    token   = os.environ.get("LINE_CHANNEL_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")
    if token and user_id:
        lines = [f"📈 kabu-watch {updated_str}", ""]
        if top:
            lines.append("【スクリーナー上位】")
            for d in top[:5]:
                ps = "+" if d["pct"] >= 0 else ""
                lines.append(f"  {d['name']} {fmt_price(d['last'])} ({ps}{d['pct']:.2f}%) score:{d['score']}")
        lines.append("")
        lines.append("【保有】")
        for d in hold_data:
            if d["last"]:
                ps = "+" if (d["pnl"] or 0) >= 0 else ""
                warn = " ⚠逆指値未設定" if d["stop_loss"] is None else ""
                lines.append(f"  {d['name']} {fmt_price(d['last'])} ({ps}{d['pnl']:.2f}%){warn}")
        send_line(token, user_id, "\n".join(lines))
        print("[OK] LINE notification sent")
    else:
        print("[INFO] LINE secrets not set — skipping notification")


if __name__ == "__main__":
    main()
