#!/usr/bin/env python3
"""Ichimoku screener for Japanese stocks — generates docs/index.html."""

import json
import os
import sys
import html as html_lib
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import requests

JST = pytz.timezone("Asia/Tokyo")


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:
        print(f"[WARN] {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def fetch_5min(symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period="5d", interval="5m", auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:
        print(f"[WARN] 5min {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


# ── Ichimoku ───────────────────────────────────────────────────────────────────

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


# ── Technical indicators ───────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.dropna().empty else float("nan")


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple:
    ema_f  = close.ewm(span=fast, min_periods=fast).mean()
    ema_s  = close.ewm(span=slow, min_periods=slow).mean()
    macd   = ema_f - ema_s
    signal = macd.ewm(span=sig, min_periods=sig).mean()
    hist   = macd - signal
    if macd.dropna().empty:
        return float("nan"), float("nan"), float("nan")
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def calc_volume_signal(df: pd.DataFrame) -> tuple:
    if len(df) < 21 or "Volume" not in df.columns:
        return None, False
    avg20   = float(df["Volume"].iloc[-21:-1].mean())
    current = float(df["Volume"].iloc[-1])
    if avg20 <= 0:
        return None, False
    ratio = current / avg20
    return ratio, ratio >= 2.0


def calc_52w_drawdown(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return float("nan")
    high_52w = float(df["High"].iloc[-min(252, len(df)):].max())
    current  = float(df["Close"].iloc[-1])
    return (current / high_52w - 1) * 100 if high_52w > 0 else float("nan")


def calc_signals(ich: pd.DataFrame, df: pd.DataFrame) -> tuple:
    empty = {
        "雲抜け": False, "三役好転": False, "上昇トレンド": False,
        "転換GC": False, "出来高急増": False,
    }
    if len(ich) < 55:
        return 0, empty

    lat = ich.iloc[-1]
    sa, sb = lat["span_a"], lat["span_b"]
    if pd.isna(sa) or pd.isna(sb):
        return 0, empty

    cloud_top = max(sa, sb)
    kumo      = bool(lat["close"] > cloud_top)
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

    _, vol_surge = calc_volume_signal(df)

    sigs = {
        "雲抜け": kumo, "三役好転": saneki, "上昇トレンド": josho,
        "転換GC": gc, "出来高急増": bool(vol_surge),
    }
    return sum(sigs.values()), sigs


# ── Analyst info ───────────────────────────────────────────────────────────────

def _parse_news_item(item: dict) -> dict:
    """Handle both old and new yfinance news formats."""
    # New format: item["content"] dict
    if "content" in item and isinstance(item["content"], dict):
        c     = item["content"]
        title = c.get("title", "")
        link  = ((c.get("canonicalUrl") or {}).get("url")
                 or (c.get("clickThroughUrl") or {}).get("url", ""))
        pub   = c.get("pubDate", "")
        if pub:
            try:
                dt  = datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(JST)
                pub = f"{dt.month}/{dt.day}"
            except Exception:
                pub = pub[:10]
        return {"title": title, "link": link, "date": pub}
    # Old format: flat dict
    title = item.get("title", "")
    link  = item.get("link", "")
    ts    = item.get("providerPublishTime", 0)
    pub   = ""
    if ts:
        try:
            pub = datetime.fromtimestamp(int(ts), tz=JST).strftime("%-m/%-d")
        except Exception:
            pass
    return {"title": title, "link": link, "date": pub}


def fetch_analyst(symbol: str, current_price: float) -> dict:
    """Return analyst target price, rating distribution, and news. Never raises."""
    out: dict = {"target": None, "target_pct": None,
                 "buy": None, "hold": None, "sell": None,
                 "rec_key": None, "news": []}
    try:
        tk   = yf.Ticker(symbol)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass

        # Average target price
        target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        if target and current_price > 0:
            out["target"]     = float(target)
            out["target_pct"] = (float(target) / current_price - 1) * 100

        out["rec_key"] = info.get("recommendationKey")

        # Buy / Hold / Sell breakdown (period-based summary)
        try:
            recs = tk.recommendations
            if recs is not None and not recs.empty:
                # Columns may be: strongBuy, buy, hold, sell, strongSell
                cols = recs.columns.str.lower()
                row  = recs.iloc[0]
                def _col(name: str) -> int:
                    for c in recs.columns:
                        if c.lower() == name:
                            return int(row[c])
                    return 0
                buy  = _col("strongbuy") + _col("buy")
                hold = _col("hold")
                sell = _col("sell") + _col("strongsell")
                if buy + hold + sell > 0:
                    out["buy"], out["hold"], out["sell"] = buy, hold, sell
        except Exception:
            pass

        # News (latest 3)
        try:
            for item in (tk.news or [])[:3]:
                parsed = _parse_news_item(item)
                if parsed.get("title"):
                    out["news"].append(parsed)
        except Exception:
            pass

    except Exception as e:
        print(f"[WARN] analyst {symbol}: {e}", file=sys.stderr)

    return out


# ── Date/time helpers ──────────────────────────────────────────────────────────

def _last_date_str(df: pd.DataFrame) -> str:
    """Return 'M/D 15:30' string for the last bar in df (daily data closes at 15:30 JST)."""
    if df.empty:
        return ""
    try:
        ts = df.index[-1]
        dt = ts.astimezone(JST) if hasattr(ts, "astimezone") else ts
        return f"{dt.month}/{dt.day} 15:30"
    except Exception:
        return ""


def _df_dates(df: pd.DataFrame, n: int) -> list:
    """Return list of date objects (or None) for last n rows, in JST."""
    try:
        return [ts.astimezone(JST).date() for ts in df.iloc[-n:].index]
    except Exception:
        try:
            return [ts.date() for ts in df.iloc[-n:].index]
        except Exception:
            return []


def _df_timestamps_jst(df5: pd.DataFrame, n: int) -> list:
    """Return list of JST datetime objects for last n rows of 5-min df."""
    try:
        return [ts.astimezone(JST) for ts in df5.iloc[-n:].index]
    except Exception:
        return []


# ── SVG helpers ────────────────────────────────────────────────────────────────

def _svg_open(w: int, h: int, rx: str = "4") -> str:
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block">'
            f'<rect width="{w}" height="{h}" fill="#0d0d1a" rx="{rx}"/>')

def _candle_color(close: float, open_: float) -> str:
    return "#ef5350" if close >= open_ else "#26a69a"

def _pct_color(pct: float) -> str:
    return "#ef5350" if pct >= 0 else "#26a69a"


# ── Price chart (日足 Ichimoku + X-axis date labels) ───────────────────────────

def make_chart(ich: pd.DataFrame, w: int = 360, h: int = 190) -> str:
    MAX_BARS = 60
    n = min(MAX_BARS, len(ich))
    if n < 3:
        return (_svg_open(w, h) +
                f'<text x="50%" y="52%" fill="#555" text-anchor="middle" font-size="11">データなし</text></svg>')

    # Extract JST dates before reset_index (for X-axis labels)
    x_dates = _df_dates(ich, n)

    d = ich.iloc[-n:].reset_index(drop=True)

    PT, PB, PL, PR = 16, 16, 4, 4   # PB=16 for date labels
    cw, ch_ = w - PL - PR, h - PT - PB

    vals = []
    for col in ["low", "high", "span_a", "span_b"]:
        vals += d[col].dropna().tolist()
    if not vals:
        return _svg_open(w, h) + "</svg>"

    pmin = min(vals) * 0.999
    pmax = max(vals) * 1.001
    prng = pmax - pmin or 1

    def py(p: float) -> float:
        return PT + ch_ * (1.0 - (p - pmin) / prng)

    slot = cw / n
    bw   = max(1.5, slot * 0.65)
    def bx(i: int) -> float:
        return PL + (i + 0.5) * slot

    c0  = float(d["close"].iloc[-1])
    c1  = float(d["close"].iloc[-2]) if len(d) > 1 else c0
    pct = (c0 / c1 - 1) * 100 if c1 else 0
    pc  = _pct_color(pct)
    lbl = f"{c0:,.0f}" if c0 >= 100 else f"{c0:.2f}"

    out = _svg_open(w, h)
    out += (f'<text x="{PL}" y="11" fill="#aaa" font-size="9" font-family="monospace">{lbl}</text>'
            f'<text x="{w - PR}" y="11" fill="{pc}" font-size="9" font-family="monospace" text-anchor="end">'
            f'{"+" if pct >= 0 else ""}{pct:.2f}%</text>')

    # Cloud polygon
    cloud_pts = [
        (i, float(d["span_a"].iloc[i]), float(d["span_b"].iloc[i]))
        for i in range(n)
        if not (pd.isna(d["span_a"].iloc[i]) or pd.isna(d["span_b"].iloc[i]))
    ]
    if len(cloud_pts) >= 2:
        top_ = " ".join(f"{bx(i):.1f},{py(max(sa,sb)):.1f}" for i, sa, sb in cloud_pts)
        bot_ = " ".join(f"{bx(i):.1f},{py(min(sa,sb)):.1f}" for i, sa, sb in reversed(cloud_pts))
        bull = sum(1 for _, sa, sb in cloud_pts if sa >= sb)
        fill = "#1a3d2a" if bull >= len(cloud_pts) / 2 else "#3d1a1a"
        out += f'<polygon points="{top_} {bot_}" fill="{fill}" opacity="0.75"/>'

    def polyline_str(col: str, stroke: str, sw: float) -> str:
        pts = [(bx(i), py(float(d[col].iloc[i]))) for i in range(n) if not pd.isna(d[col].iloc[i])]
        if len(pts) < 2:
            return ""
        return (f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                f'fill="none" stroke="{stroke}" stroke-width="{sw}" opacity="0.8"/>')

    out += polyline_str("span_a", "#26a69a", 0.8)
    out += polyline_str("span_b", "#ef5350", 0.8)
    out += polyline_str("tenkan", "#4499ff", 1.2)
    out += polyline_str("kijun",  "#ff9944", 1.2)

    # Candlesticks
    for i in range(n):
        row = d.iloc[i]
        o_, c_, hh, ll = row["open"], row["close"], row["high"], row["low"]
        if any(pd.isna(v) for v in [o_, c_, hh, ll]):
            continue
        x_  = bx(i)
        col = _candle_color(float(c_), float(o_))
        yt  = py(max(float(o_), float(c_)))
        yb  = py(min(float(o_), float(c_)))
        out += (f'<line x1="{x_:.1f}" y1="{py(float(hh)):.1f}" x2="{x_:.1f}" y2="{py(float(ll)):.1f}" '
                f'stroke="{col}" stroke-width="0.8"/>'
                f'<rect x="{x_ - bw/2:.1f}" y="{yt:.1f}" width="{bw:.1f}" '
                f'height="{max(1.0, yb - yt):.1f}" fill="{col}"/>')

    # X-axis date labels (every ~10 bars, always include last)
    if x_dates:
        last_lx = -999.0
        indices = list(range(0, n, max(1, n // 6))) + [n - 1]
        seen = set()
        for i in sorted(set(indices)):
            if i >= len(x_dates) or i in seen:
                continue
            seen.add(i)
            x_ = bx(i)
            if x_ - last_lx < 32:
                continue
            dt  = x_dates[i]
            lbl_d = f"{dt.month}/{dt.day}"
            out += (f'<text x="{x_:.1f}" y="{h - 3}" fill="#556" font-size="8" '
                    f'font-family="monospace" text-anchor="middle">{lbl_d}</text>')
            last_lx = x_

    return out + "</svg>"


# ── Volume bar chart ───────────────────────────────────────────────────────────

def make_volume_chart(df: pd.DataFrame, w: int = 360, h: int = 55, max_bars: int = 60) -> str:
    if "Volume" not in df.columns or len(df) < 3:
        return ""
    avg20 = float(df["Volume"].iloc[-21:-1].mean()) if len(df) >= 21 else float(df["Volume"].mean())

    n = min(max_bars, len(df))
    d = df.iloc[-n:].reset_index(drop=True)
    vol  = d["Volume"].astype(float)
    vmax = vol.max()
    if vmax == 0 or pd.isna(vmax):
        return ""

    PL, PR, PT, PB = 4, 4, 4, 2
    cw, ch_ = w - PL - PR, h - PT - PB
    slot = cw / n
    bw   = max(1.5, slot * 0.65)
    def bx(i: int) -> float: return PL + (i + 0.5) * slot

    out = _svg_open(w, h, rx="0")
    if avg20 > 0 and avg20 <= vmax:
        ay = PT + ch_ * (1 - avg20 / vmax)
        out += (f'<line x1="{PL}" y1="{ay:.1f}" x2="{w-PR}" y2="{ay:.1f}" '
                f'stroke="#555" stroke-width="0.8" stroke-dasharray="3,2"/>')

    last_v = int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
    for i in range(n):
        v = vol.iloc[i]
        if pd.isna(v) or v == 0:
            continue
        bh_  = ch_ * v / vmax
        by_  = PT + ch_ - bh_
        col  = _candle_color(float(d["Close"].iloc[i]), float(d["Open"].iloc[i]))
        if avg20 > 0 and v >= avg20 * 2:
            col = "#ffd600"
        out += (f'<rect x="{bx(i)-bw/2:.1f}" y="{by_:.1f}" width="{bw:.1f}" '
                f'height="{max(1.0, bh_):.1f}" fill="{col}" opacity="0.85"/>')

    lbl_v = (f"{last_v/1_000_000:.1f}M" if last_v >= 1_000_000
             else f"{last_v/1_000:.0f}K" if last_v >= 1_000 else str(last_v))
    out += f'<text x="{PL}" y="{h-2}" fill="#666" font-size="8" font-family="monospace">vol {lbl_v}</text>'
    return out + "</svg>"


# ── 5-minute chart (ローソク足 + X-axis time labels) ─────────────────────────

def make_5min_chart(df5: pd.DataFrame, w: int = 360, h: int = 190) -> str:
    MAX_BARS = 160
    n = min(MAX_BARS, len(df5))
    if n < 3:
        return (_svg_open(w, h) +
                '<text x="50%" y="52%" fill="#555" text-anchor="middle" font-size="11">データなし</text></svg>')

    slice_     = df5.iloc[-n:]
    x_times    = _df_timestamps_jst(df5, n)   # JST datetimes
    day_boundaries: set = set()
    if x_times:
        for i in range(1, len(x_times)):
            if x_times[i].date() != x_times[i - 1].date():
                day_boundaries.add(i)

    d = slice_.reset_index(drop=True)

    PT, PB, PL, PR = 16, 16, 4, 4   # PB=16 for time labels
    cw, ch_ = w - PL - PR, h - PT - PB
    pmin = float(d["Low"].min()) * 0.999
    pmax = float(d["High"].max()) * 1.001
    prng = pmax - pmin or 1

    def py(p: float) -> float: return PT + ch_ * (1 - (p - pmin) / prng)
    slot = cw / n
    bw   = max(0.8, slot * 0.7)
    def bx(i: int) -> float: return PL + (i + 0.5) * slot

    c0  = float(d["Close"].iloc[-1])
    c1  = float(d["Close"].iloc[-2]) if len(d) > 1 else c0
    pct = (c0 / c1 - 1) * 100 if c1 else 0
    pc  = _pct_color(pct)
    lbl = f"{c0:,.0f}" if c0 >= 100 else f"{c0:.2f}"

    out = _svg_open(w, h)
    out += (f'<text x="{PL}" y="11" fill="#aaa" font-size="9" font-family="monospace">{lbl}</text>'
            f'<text x="{w//2}" y="11" fill="#444" font-size="8" font-family="monospace" text-anchor="middle">5分足</text>'
            f'<text x="{w-PR}" y="11" fill="{pc}" font-size="9" font-family="monospace" text-anchor="end">'
            f'{"+" if pct >= 0 else ""}{pct:.2f}%</text>')

    # Day separator lines
    for i in day_boundaries:
        if 0 < i < n:
            xsep = bx(i) - slot / 2
            out += f'<line x1="{xsep:.1f}" y1="{PT}" x2="{xsep:.1f}" y2="{h-PB}" stroke="#2a2a44" stroke-width="1"/>'

    # Candlesticks
    for i in range(n):
        row = d.iloc[i]
        o_, c_, hh, ll = row["Open"], row["Close"], row["High"], row["Low"]
        if any(pd.isna(v) for v in [o_, c_, hh, ll]):
            continue
        x_  = bx(i)
        col = _candle_color(float(c_), float(o_))
        yt  = py(max(float(o_), float(c_)))
        yb  = py(min(float(o_), float(c_)))
        out += (f'<line x1="{x_:.1f}" y1="{py(float(hh)):.1f}" x2="{x_:.1f}" y2="{py(float(ll)):.1f}" '
                f'stroke="{col}" stroke-width="0.5"/>'
                f'<rect x="{x_-bw/2:.1f}" y="{yt:.1f}" width="{bw:.1f}" '
                f'height="{max(0.8, yb - yt):.1f}" fill="{col}"/>')

    # X-axis time labels: day boundaries + key hours (09, 12, 15)
    if x_times:
        last_lx = -999.0
        for i, ts in enumerate(x_times):
            is_boundary = i in day_boundaries
            is_key_hour = ts.minute == 0 and ts.hour in (9, 12, 15)
            if not (is_boundary or is_key_hour):
                continue
            x_ = bx(i)
            if x_ - last_lx < 33:
                continue
            if is_boundary:
                lbl_t = f"{ts.month}/{ts.day}"
            else:
                lbl_t = f"{ts.hour:02d}:00"
            out += (f'<text x="{x_:.1f}" y="{h - 3}" fill="#556" font-size="8" '
                    f'font-family="monospace" text-anchor="middle">{lbl_t}</text>')
            last_lx = x_

    return out + "</svg>"


# ── HTML helpers ───────────────────────────────────────────────────────────────

SIGNAL_META = {
    "雲抜け":      {"color": "#2979ff", "bg": "#0d2a66"},
    "三役好転":    {"color": "#ffd600", "bg": "#4a3a00"},
    "上昇トレンド": {"color": "#00e676", "bg": "#003322"},
    "転換GC":      {"color": "#ea80fc", "bg": "#330044"},
    "出来高急増":  {"color": "#ffd600", "bg": "#3d2800"},
}


def signal_badges(sigs: dict) -> str:
    badges = [
        f'<span style="background:{m["bg"]};color:{m["color"]};border:1px solid {m["color"]};'
        f'border-radius:4px;padding:2px 6px;font-size:11px;white-space:nowrap">{name}</span>'
        for name, active in sigs.items()
        if active and (m := SIGNAL_META.get(name, {"color": "#aaa", "bg": "#222"}))
    ]
    return " ".join(badges) if badges else '<span style="color:#555;font-size:11px">シグナルなし</span>'


def fmt_price(p: float) -> str:
    return f"{p:,.0f}" if p >= 100 else f"{p:.2f}"


def card(content: str) -> str:
    return f'<div style="background:#111122;border-radius:8px;padding:12px;margin-bottom:12px">{content}</div>'


def safe_id(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "X")


def chart_toggle(sym: str, day_price: str, day_vol: str, m5_price: str, m5_vol: str) -> str:
    sid    = safe_id(sym)
    has_m5 = bool(m5_price)
    extra  = (f'<button id="b5_{sid}" onclick="sw(\'{sid}\',\'m\')" class="tab">5分足</button>'
              if has_m5 else "")
    m5_div = (f'<div id="c5_{sid}" style="display:none">{m5_price}{m5_vol}</div>'
              if has_m5 else "")
    return (f'<div style="display:flex;gap:4px;margin-bottom:6px">'
            f'<button id="bd_{sid}" onclick="sw(\'{sid}\',\'d\')" class="tab active">日足</button>'
            f'{extra}</div>'
            f'<div id="cd_{sid}">{day_price}{day_vol}</div>'
            f'{m5_div}')


def stats_row(rsi: float, macd_v: float, macd_s: float, macd_h: float,
              drawdown: float, vol_ratio) -> str:
    parts = []

    if not pd.isna(rsi):
        rc   = "#ef5350" if rsi >= 70 else "#26a69a" if rsi <= 30 else "#aaa"
        note = (" <span style='color:#ef5350;font-size:10px'>過熱</span>" if rsi >= 70
                else " <span style='color:#26a69a;font-size:10px'>売られすぎ</span>" if rsi <= 30 else "")
        parts.append(f'<span>RSI <span style="color:{rc}">{rsi:.1f}</span>{note}</span>')

    if not (pd.isna(macd_v) or pd.isna(macd_h)):
        hc  = "#ef5350" if macd_h >= 0 else "#26a69a"
        mv  = f"{macd_v:.2f}" if abs(macd_v) < 1000 else f"{macd_v:.0f}"
        hv  = f'{"+" if macd_h >= 0 else ""}{macd_h:.2f}' if abs(macd_h) < 1000 else f'{macd_h:.0f}'
        parts.append(f'<span>MACD <span style="color:#999">{mv}</span> '
                     f'<span style="color:{hc}">({hv})</span></span>')

    if not pd.isna(drawdown):
        dc = "#ef5350" if drawdown >= -5 else "#ff9944" if drawdown >= -20 else "#26a69a"
        parts.append(f'<span>高値比 <span style="color:{dc}">{drawdown:+.1f}%</span></span>')

    if vol_ratio is not None and not pd.isna(vol_ratio):
        vc = "#ffd600" if vol_ratio >= 2.0 else "#666"
        parts.append(f'<span>出来高 <span style="color:{vc}">{vol_ratio:.1f}×</span></span>')

    if not parts:
        return ""
    return ('<div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#666;'
            'margin:5px 0 7px;border-top:1px solid #1a1a3a;padding-top:5px">'
            + " ".join(parts) + "</div>")


def render_analyst_section(analyst: dict) -> str:
    """Render analyst target price, rating distribution, and news."""
    parts = []

    # Target price
    if analyst.get("target") is not None and analyst.get("target_pct") is not None:
        tp   = analyst["target"]
        tpct = analyst["target_pct"]
        tc   = _pct_color(tpct)
        sign = "+" if tpct >= 0 else ""
        parts.append(
            f'<div style="font-size:11px;margin-bottom:4px">'
            f'目標株価 <span style="color:#ffd600">{fmt_price(tp)}</span>'
            f' <span style="color:{tc}">({sign}{tpct:.1f}%)</span></div>'
        )

    # Buy / Hold / Sell distribution
    buy, hold, sell = analyst.get("buy"), analyst.get("hold"), analyst.get("sell")
    if buy is not None or hold is not None or sell is not None:
        total = (buy or 0) + (hold or 0) + (sell or 0)
        if total > 0:
            parts.append(
                f'<div style="display:flex;gap:10px;font-size:11px;margin-bottom:4px">'
                f'<span style="color:#ef5350">▲Buy {buy or 0}</span>'
                f'<span style="color:#888">━Hold {hold or 0}</span>'
                f'<span style="color:#26a69a">▼Sell {sell or 0}</span>'
                f'</div>'
            )
    elif analyst.get("rec_key"):
        rec_map = {
            "strong_buy": ("Strong Buy", "#ef5350"), "buy": ("Buy", "#ef5350"),
            "hold": ("Hold", "#888"),
            "sell": ("Sell", "#26a69a"), "strong_sell": ("Strong Sell", "#26a69a"),
        }
        rl, rc = rec_map.get(analyst["rec_key"], (analyst["rec_key"], "#888"))
        parts.append(f'<div style="font-size:11px;margin-bottom:4px">'
                     f'コンセンサス <span style="color:{rc}">{rl}</span></div>')

    # News
    news_items = analyst.get("news", [])
    if news_items:
        news_html = ""
        for item in news_items:
            raw_title = item.get("title", "")
            title     = html_lib.escape(raw_title[:55] + ("…" if len(raw_title) > 55 else ""))
            link      = html_lib.escape(item.get("link", "#"))
            date      = html_lib.escape(item.get("date", ""))
            date_span = f' <span style="color:#555;font-size:10px">{date}</span>' if date else ""
            news_html += (
                f'<div style="margin-bottom:3px;line-height:1.3">'
                f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#7c83ff;text-decoration:none;font-size:11px">{title}</a>'
                f'{date_span}</div>'
            )
        parts.append(f'<div style="margin-top:2px">{news_html}</div>')

    if not parts:
        return '<div style="font-size:11px;color:#555;margin-top:6px;padding-top:6px;border-top:1px solid #1a1a3a">評価データなし</div>'

    return ('<div style="border-top:1px solid #1a1a3a;margin-top:8px;padding-top:8px">'
            + "".join(parts) + "</div>")


# ── LINE ───────────────────────────────────────────────────────────────────────

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


# ── Per-symbol data builder ────────────────────────────────────────────────────

def build_data(symbol: str, name: str) -> dict:
    df  = fetch_ohlcv(symbol, period="1y")
    df5 = fetch_5min(symbol)

    base: dict = {"symbol": symbol, "name": name, "error": True,
                  "last": None, "pct": 0.0, "score": 0, "sigs": {},
                  "last_date_str": "", "analyst": {"news": []}}
    if len(df) < 55:
        return base

    ich              = ichimoku(df)
    score, sigs      = calc_signals(ich, df)
    rsi_v            = calc_rsi(df["Close"])
    macd_v, macd_s, macd_h = calc_macd(df["Close"])
    vol_ratio, _     = calc_volume_signal(df)
    drawdown         = calc_52w_drawdown(df)
    last             = float(df["Close"].iloc[-1])
    prev             = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    pct              = (last / prev - 1) * 100 if prev else 0.0
    last_date_str    = _last_date_str(df)
    analyst          = fetch_analyst(symbol, last)

    day_price = make_chart(ich)
    day_vol   = make_volume_chart(df, max_bars=60)
    m5_price  = make_5min_chart(df5) if len(df5) >= 5 else ""
    m5_vol    = make_volume_chart(df5, max_bars=160) if len(df5) >= 5 else ""

    return {
        "symbol": symbol, "name": name, "error": False,
        "last": last, "pct": pct, "score": score, "sigs": sigs,
        "rsi": rsi_v, "macd_v": macd_v, "macd_s": macd_s, "macd_h": macd_h,
        "vol_ratio": vol_ratio, "drawdown": drawdown,
        "last_date_str": last_date_str, "analyst": analyst,
        "day_price": day_price, "day_vol": day_vol,
        "m5_price": m5_price, "m5_vol": m5_vol,
    }


# ── Card renderers ─────────────────────────────────────────────────────────────

def _stats(d: dict) -> str:
    return stats_row(d.get("rsi", float("nan")), d.get("macd_v", float("nan")),
                     d.get("macd_s", float("nan")), d.get("macd_h", float("nan")),
                     d.get("drawdown", float("nan")), d.get("vol_ratio"))


def _toggle(d: dict) -> str:
    return chart_toggle(d["symbol"],
                        d.get("day_price", ""), d.get("day_vol", ""),
                        d.get("m5_price", ""),  d.get("m5_vol", ""))


def _data_ts(d: dict) -> str:
    ds = d.get("last_date_str", "")
    return f'<span style="font-size:10px;color:#555">{ds}時点</span>' if ds else ""


def render_index_card(d: dict) -> str:
    if d.get("error") or d["last"] is None:
        return card(f'<div style="color:#888">{d["name"]} — データ取得失敗</div>')
    pc = _pct_color(d["pct"])
    ps = "+" if d["pct"] >= 0 else ""
    return card(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
        f'<span style="font-weight:600;font-size:14px">{d["name"]}</span>'
        f'<div style="text-align:right"><span style="font-size:11px;color:#666">{d["symbol"]}</span><br>'
        f'{_data_ts(d)}</div></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
        f'<span style="font-size:20px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</span>'
        f'<span style="font-size:14px;color:{pc}">{ps}{d["pct"]:.2f}%</span></div>'
        + _stats(d) + _toggle(d) + render_analyst_section(d.get("analyst", {}))
    )


def render_holding_card(d: dict) -> str:
    if d.get("error") or d["last"] is None:
        return card(f'<div style="color:#888">{d["name"]} — データ取得失敗</div>')
    pnl = d.get("pnl", 0.0) or 0.0
    pc  = _pct_color(pnl)
    ps  = "+" if pnl >= 0 else ""
    warn = ""
    if d.get("stop_loss") is None:
        warn = ('<div style="background:#3a1a00;border:1px solid #ff6600;border-radius:4px;'
                'padding:4px 8px;margin-top:6px;font-size:11px;color:#ff9944">⚠ 逆指値未設定</div>')
    return card(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
        f'<span style="font-weight:600;font-size:14px">{d["name"]}</span>'
        f'<div style="text-align:right"><span style="font-size:11px;color:#666">{d["symbol"]}</span><br>'
        f'{_data_ts(d)}</div></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
        f'<span style="font-size:18px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</span>'
        f'<span style="font-size:14px;color:{pc}">{ps}{pnl:.2f}%</span></div>'
        f'<div style="font-size:11px;color:#666;margin-bottom:4px">'
        f'取得 {fmt_price(d["cost"])} 円'
        + (f' | 逆指値 {fmt_price(d["stop_loss"])} 円' if d.get("stop_loss") else "")
        + f'</div>'
        + (f'<div style="margin-bottom:6px">{signal_badges(d["sigs"])}</div>' if d.get("sigs") else "")
        + _stats(d) + _toggle(d) + warn + render_analyst_section(d.get("analyst", {}))
    )


def render_candidate_card(d: dict) -> str:
    sc_colors = ["#333", "#1a3d2a", "#1a4a2a", "#1a5a2a", "#007744", "#00aa55"]
    pc  = _pct_color(d["pct"])
    ps  = "+" if d["pct"] >= 0 else ""
    sc  = sc_colors[min(d["score"], 5)]
    return card(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="background:{sc};color:#fff;border-radius:4px;padding:2px 7px;'
        f'font-size:13px;font-weight:700;min-width:22px;text-align:center">{d["score"]}</span>'
        f'<div><div style="font-weight:600;font-size:14px">{d["name"]}</div>'
        f'<div style="display:flex;gap:6px;align-items:center">'
        f'<span style="font-size:11px;color:#666">{d["symbol"]}</span>'
        f'{_data_ts(d)}</div></div></div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:16px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</div>'
        f'<div style="font-size:12px;color:{pc}">{ps}{d["pct"]:.2f}%</div></div></div>'
        f'<div style="margin-bottom:6px">{signal_badges(d["sigs"])}</div>'
        + _stats(d) + _toggle(d) + render_analyst_section(d.get("analyst", {}))
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    now_jst = datetime.now(JST)
    os.makedirs("docs", exist_ok=True)

    idx_rows = [build_data(i["symbol"], i["name"]) for i in cfg["indices"]]

    hold_rows = []
    for h in cfg["holdings"]:
        d = build_data(h["symbol"], h["name"])
        d["cost"]      = h["cost"]
        d["stop_loss"] = h.get("stop_loss")
        d["pnl"]       = (d["last"] / h["cost"] - 1) * 100 if d["last"] else None
        hold_rows.append(d)

    max_n     = cfg.get("max_candidates", 10)
    cand_rows = [build_data(c["symbol"], c["name"]) for c in cfg["candidates"]]
    cand_rows = [d for d in cand_rows if not d["error"]]
    cand_rows.sort(key=lambda x: (-x["score"], -x["pct"]))
    top = cand_rows[:max_n]

    # データ取得時刻: use the last date from any index row
    data_date_str = next((d["last_date_str"] for d in idx_rows if d.get("last_date_str")), "")
    run_str       = now_jst.strftime("%-m/%-d %H:%M")
    header_note   = (f'データ: {data_date_str}時点 | 更新: {run_str} JST'
                     if data_date_str else f'更新: {run_str} JST')

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
.tab{{background:#1a1a2e;color:#777;border:1px solid #2a2a44;border-radius:4px;
  padding:3px 10px;font-size:11px;cursor:pointer}}
.tab.active{{background:#2a2a4e;color:#e0e0f0;border-color:#7c83ff}}
</style>
<script>
function sw(id,t){{
  var d=document.getElementById('cd_'+id),m=document.getElementById('c5_'+id);
  var bd=document.getElementById('bd_'+id),bm=document.getElementById('b5_'+id);
  if(d)d.style.display=t==='d'?'block':'none';
  if(m)m.style.display=t==='d'?'none':'block';
  if(bd)bd.className=t==='d'?'tab active':'tab';
  if(bm)bm.className=t==='d'?'tab':'tab active';
}}
</script>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0 10px">
  <div style="font-size:20px;font-weight:800;color:#7c83ff">kabu-watch</div>
  <div style="font-size:10px;color:#555;text-align:right;line-height:1.5">{header_note}</div>
</div>

<h2>マーケット</h2>
{"".join(render_index_card(d) for d in idx_rows)}

<h2>保有銘柄</h2>
{"".join(render_holding_card(d) for d in hold_rows)}

<h2>スクリーナー TOP{len(top)}</h2>
<div style="font-size:11px;color:#555;margin-bottom:8px">スコア = 雲抜け+三役好転+上昇トレンド+転換GC+出来高急増 (最大5)</div>
{"".join(render_candidate_card(d) for d in top) or card('<div style="color:#888;text-align:center">候補銘柄なし</div>')}

<div style="text-align:center;font-size:10px;color:#333;padding:16px 0 8px">
  自動更新: 平日 16:30 JST | <a href="https://github.com/yousukekomai6150-sketch/kabu-watch" style="color:#444">GitHub</a>
</div>
</body>
</html>"""

    out_path = "docs/index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {out_path} written ({len(html):,} bytes)")

    # LINE
    token   = os.environ.get("LINE_CHANNEL_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")
    if token and user_id:
        lines = [f"📈 kabu-watch {run_str} JST", ""]
        if top:
            lines.append("【スクリーナー上位】")
            for d in top[:5]:
                ps = "+" if d["pct"] >= 0 else ""
                lines.append(f"  {d['name']} {fmt_price(d['last'])} ({ps}{d['pct']:.2f}%) score:{d['score']}")
        lines += ["", "【保有】"]
        for d in hold_rows:
            if d["last"]:
                pnl  = d.get("pnl") or 0
                ps   = "+" if pnl >= 0 else ""
                warn = " ⚠逆指値未設定" if d.get("stop_loss") is None else ""
                lines.append(f"  {d['name']} {fmt_price(d['last'])} ({ps}{pnl:.2f}%){warn}")
        send_line(token, user_id, "\n".join(lines))
        print("[OK] LINE notification sent")
    else:
        print("[INFO] LINE secrets not set — skipping notification")


if __name__ == "__main__":
    main()
