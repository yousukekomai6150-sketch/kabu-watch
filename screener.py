#!/usr/bin/env python3
"""Ichimoku screener for Japanese stocks — generates docs/index.html."""

import json
import os
import sys
import html as html_lib
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime as _parse_rfc2822
import pytz
import yfinance as yf
import pandas as pd
import requests
import re

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


def fetch_1min(symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m", auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:
        print(f"[WARN] 1min {symbol}: {e}", file=sys.stderr)
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


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return float("nan")
    h  = df["High"]
    l  = df["Low"]
    pc = df["Close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    v = atr.iloc[-1]
    return float(v) if not pd.isna(v) else float("nan")


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

def _kabutan_news_url(symbol: str) -> str:
    """kabutan.jp RSS URL を返す。指数・非.T銘柄はマーケットニュース。"""
    if not symbol.endswith(".T"):
        return "https://kabutan.jp/news/rss/?category=market"
    code = symbol[:-2]   # "7012.T" → "7012"
    return f"https://kabutan.jp/news/rss/?code={code}"


def fetch_kabutan_news(symbol: str, max_items: int = 3) -> list:
    """kabutan.jp RSS から最新ニュースを取得。取得失敗時は [] を返す。"""
    url = _kabutan_news_url(symbol)
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 kabu-watch/1.0"},
        )
        resp.raise_for_status()
        root  = ET.fromstring(resp.content)
        items = []
        for elem in root.findall(".//item")[:max_items]:
            title = (elem.findtext("title") or "").strip()
            link  = (elem.findtext("link")  or "").strip()
            pub   = (elem.findtext("pubDate") or "").strip()
            date_str = ""
            if pub:
                try:
                    dt = _parse_rfc2822(pub).astimezone(JST)
                    date_str = f"{dt.month}/{dt.day}"
                except Exception:
                    pass
            if title:
                items.append({"title": title, "link": link, "date": date_str})
        return items
    except Exception as e:
        print(f"[WARN] kabutan news {symbol}: {e}", file=sys.stderr)
        return []


def fetch_surge_from_kabutan(max_items: int = 60) -> list:
    """kabutan.jp 値上がり率ランキングから .T シンボルリストを取得。失敗時は []。"""
    url = "https://kabutan.jp/warning/?mode=2&market=0"
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 kabu-watch/1.0"})
        resp.raise_for_status()
        codes = re.findall(r'/stock/\?code=(\d{4})', resp.text)
        seen, symbols = set(), []
        for c in codes:
            if c not in seen:
                seen.add(c)
                symbols.append(f"{c}.T")
                if len(symbols) >= max_items:
                    break
        return symbols
    except Exception as e:
        print(f"[WARN] kabutan surge fetch: {e}", file=sys.stderr)
        return []


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

        # News from kabutan.jp RSS (日本語)
        out["news"] = fetch_kabutan_news(symbol)

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

    # X-axis date labels: first bar of each ISO week (≈週1本), always include first/last
    if x_dates:
        label_idxs = {0, min(n - 1, len(x_dates) - 1)}
        for i in range(1, min(n, len(x_dates))):
            if x_dates[i].isocalendar()[1] != x_dates[i - 1].isocalendar()[1]:
                label_idxs.add(i)
        last_lx = -999.0
        for i in sorted(label_idxs):
            if i >= len(x_dates):
                continue
            x_ = bx(i)
            if x_ - last_lx < 30:   # skip if too close (handles dense edge cases)
                continue
            dt = x_dates[i]
            out += (f'<text x="{x_:.1f}" y="{h - 3}" fill="#ccd6e0" font-size="8" '
                    f'font-family="monospace" text-anchor="middle">{dt.month}/{dt.day}</text>')
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

def make_intraday_chart(df5: pd.DataFrame, max_bars: int = 160,
                        interval_label: str = "5分足", w: int = 360, h: int = 190) -> str:
    n = min(max_bars, len(df5))
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
            f'<text x="{w//2}" y="11" fill="#444" font-size="8" font-family="monospace" text-anchor="middle">{interval_label}</text>'
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

    # X-axis time labels: 30分刻み (9:00, 9:30, 10:00...) within TSE trading hours
    # min_spacing=25px → 2日表示では自動的に1時間刻みに間引かれる
    if x_times:
        last_lx = -999.0
        for i, ts in enumerate(x_times):
            is_boundary = i in day_boundaries
            is_30min    = ts.minute in (0, 30) and 9 <= ts.hour <= 15
            if not (is_boundary or is_30min):
                continue
            x_ = bx(i)
            if x_ - last_lx < 25:
                continue
            lbl_t = (f"{ts.month}/{ts.day}" if is_boundary
                     else f"{ts.hour}:{ts.minute:02d}")
            out += (f'<text x="{x_:.1f}" y="{h - 3}" fill="#ccd6e0" font-size="8" '
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


def _market_cap_str(cap) -> str:
    if cap is None:
        return "不明"
    oku = cap / 1e8
    return f"{oku/10000:.1f}兆" if oku >= 10000 else f"{oku:.0f}億"


def card(content: str) -> str:
    return f'<div style="background:#111122;border-radius:8px;padding:12px;margin-bottom:12px">{content}</div>'


def safe_id(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "X")


def chart_toggle(sym: str, day_price: str, day_vol: str,
                 m5_price: str, m5_vol: str,
                 m1_price: str = "", m1_vol: str = "") -> str:
    sid    = safe_id(sym)
    has_m5 = bool(m5_price)
    has_m1 = bool(m1_price)
    tabs   = (f'<button id="bd_{sid}" onclick="sw(\'{sid}\',\'d\')" class="tab active">日足</button>')
    if has_m5:
        tabs += f'<button id="b5_{sid}" onclick="sw(\'{sid}\',\'m\')" class="tab">5分足</button>'
    if has_m1:
        tabs += f'<button id="b1_{sid}" onclick="sw(\'{sid}\',\'1\')" class="tab">1分足</button>'
    divs   = f'<div id="cd_{sid}">{day_price}{day_vol}</div>'
    if has_m5:
        divs  += f'<div id="c5_{sid}" style="display:none">{m5_price}{m5_vol}</div>'
    if has_m1:
        divs  += f'<div id="c1_{sid}" style="display:none">{m1_price}{m1_vol}</div>'
    return (f'<div style="display:flex;gap:4px;margin-bottom:6px">{tabs}</div>'
            + divs)


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

    # News from kabutan.jp (always shown)
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
        parts.append(f'<div style="margin-top:4px">{news_html}</div>')
    else:
        parts.append('<div style="font-size:11px;color:#555;margin-top:4px">ニュースなし</div>')

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
    base: dict = {"symbol": symbol, "name": name, "error": True, "error_msg": "データ取得失敗",
                  "last": None, "pct": 0.0, "score": 0, "sigs": {},
                  "last_date_str": "", "analyst": {"news": []}}
    if "=F" in symbol:
        base["error_msg"] = "先物データ取得不可"
        return base
    try:
        df  = fetch_ohlcv(symbol, period="1y")
        df5 = fetch_5min(symbol)
        df1 = fetch_1min(symbol)

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
        m5_price  = make_intraday_chart(df5, max_bars=160, interval_label="5分足") if len(df5) >= 5 else ""
        m5_vol    = make_volume_chart(df5, max_bars=160) if len(df5) >= 5 else ""
        m1_price  = make_intraday_chart(df1, max_bars=200, interval_label="1分足") if len(df1) >= 5 else ""
        m1_vol    = make_volume_chart(df1, max_bars=200) if len(df1) >= 5 else ""

        # 逆指値推奨・押し目レンジ (一目均衡表ベース)
        stop_recommend = None
        dip_buy_range  = None
        lat = ich.iloc[-1]
        kijun_v  = float(lat["kijun"])  if not pd.isna(lat["kijun"])  else None
        tenkan_v = float(lat["tenkan"]) if not pd.isna(lat["tenkan"]) else None
        span_a_v = float(lat["span_a"]) if not pd.isna(lat["span_a"]) else None
        span_b_v = float(lat["span_b"]) if not pd.isna(lat["span_b"]) else None
        cloud_bot_v = (min(span_a_v, span_b_v)
                       if span_a_v is not None and span_b_v is not None else None)
        cloud_top_v = (max(span_a_v, span_b_v)
                       if span_a_v is not None and span_b_v is not None else None)

        # ① 基準線・雲下限のうち現在値より低い候補の最小値
        sr_candidates = [v for v in [kijun_v, cloud_bot_v]
                         if v is not None and v < last]
        if sr_candidates:
            stop_recommend = min(sr_candidates)
        else:
            # ② 両方が現在値以上(雲下・トレンド崩れ) → ATR(14)×1.5 フォールバック
            atr_v = calc_atr(df)
            if not pd.isna(atr_v) and atr_v > 0:
                stop_recommend = last - atr_v * 1.5

        # 押し目レンジ: 上昇シグナルあり かつ 雲の中にいない場合のみ
        if kijun_v is not None and tenkan_v is not None \
                and cloud_bot_v is not None and cloud_top_v is not None:
            in_cloud = cloud_bot_v <= last <= cloud_top_v
            if (sigs.get("雲抜け") or sigs.get("上昇トレンド")) and not in_cloud:
                dip_buy_range = (min(tenkan_v, kijun_v), max(tenkan_v, kijun_v))

        return {
            "symbol": symbol, "name": name, "error": False,
            "last": last, "pct": pct, "score": score, "sigs": sigs,
            "rsi": rsi_v, "macd_v": macd_v, "macd_s": macd_s, "macd_h": macd_h,
            "vol_ratio": vol_ratio, "drawdown": drawdown,
            "last_date_str": last_date_str, "analyst": analyst,
            "stop_recommend": stop_recommend, "dip_buy_range": dip_buy_range,
            "day_price": day_price, "day_vol": day_vol,
            "m5_price": m5_price, "m5_vol": m5_vol,
            "m1_price": m1_price, "m1_vol": m1_vol,
        }
    except Exception as e:
        print(f"[WARN] build_data {symbol}: {e}", file=sys.stderr)
        return base


# ── Surge screening data builder ─────────────────────────────────────────────

def build_surge_data(symbol: str, name: str) -> dict | None:
    """急騰スクリーニング用軽量ビルダー。フィルター不通過は None を返す。"""
    try:
        df = fetch_ohlcv(symbol, period="3mo")
        if len(df) < 21:
            return None

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        pct  = (last / prev - 1) * 100 if prev else 0.0
        vol_ratio, _ = calc_volume_signal(df)

        # 価格・騰落率・出来高の早期フィルター
        if last > 5000 or pct < 3.0 or vol_ratio is None or vol_ratio < 3.0:
            return None

        # フィルター通過後のみ時価総額を取得（API呼び出し節約）
        market_cap = None
        try:
            info = yf.Ticker(symbol).info or {}
            market_cap = info.get("marketCap")
            # 名称を yfinance から補完（kabutan 由来でコードのみの場合）
            if name == symbol.replace(".T", ""):
                name = (info.get("longName") or info.get("shortName") or name)[:20]
        except Exception:
            pass

        if market_cap is not None and market_cap > 50_000_000_000:
            return None

        rsi_v = calc_rsi(df["Close"])
        return {
            "symbol": symbol, "name": name,
            "last": last, "pct": pct,
            "vol_ratio": vol_ratio, "rsi": rsi_v,
            "market_cap": market_cap,
            "surge_score": vol_ratio * pct,
        }
    except Exception as e:
        print(f"[WARN] surge {symbol}: {e}", file=sys.stderr)
        return None


# ── Card renderers ─────────────────────────────────────────────────────────────

def _stats(d: dict) -> str:
    return stats_row(d.get("rsi", float("nan")), d.get("macd_v", float("nan")),
                     d.get("macd_s", float("nan")), d.get("macd_h", float("nan")),
                     d.get("drawdown", float("nan")), d.get("vol_ratio"))


def _toggle(d: dict) -> str:
    return chart_toggle(d["symbol"],
                        d.get("day_price", ""), d.get("day_vol", ""),
                        d.get("m5_price", ""),  d.get("m5_vol", ""),
                        d.get("m1_price", ""),  d.get("m1_vol", ""))


def _data_ts(d: dict) -> str:
    ds = d.get("last_date_str", "")
    return f'<span style="font-size:10px;color:#555">{ds}時点</span>' if ds else ""


def _stop_recommend_html(d: dict) -> str:
    sr   = d.get("stop_recommend")
    last = d.get("last")
    if sr is None or last is None or sr <= 0 or sr >= last:
        return ""
    pct = (sr / last - 1) * 100   # always negative
    return (
        f'<div style="background:#0e1a2e;border:1px solid #1e3a5e;border-radius:4px;'
        f'padding:5px 8px;margin:4px 0;font-size:11px">'
        f'<span style="color:#4499ff">⊙ 逆指値推奨</span>  '
        f'<span style="color:#e0e0f0;font-weight:600">{fmt_price(sr)}</span>円  '
        f'<span style="color:#26a69a">({pct:+.1f}%)</span>'
        f'<span style="color:#555;font-size:10px">  基準線・雲下限の低い方</span>'
        f'</div>'
    )


def _dip_buy_html(d: dict) -> str:
    dbr  = d.get("dip_buy_range")
    last = d.get("last")
    if dbr is None or last is None:
        return ""
    low, high = dbr
    if low >= last:
        return ""
    low_pct  = (low  / last - 1) * 100
    high_pct = (high / last - 1) * 100
    if abs(high - low) < last * 0.001:
        return (f'<div style="font-size:11px;color:#7c83ff;margin:3px 0 5px">'
                f'📍 押し目 <span style="color:#e0e0f0">{fmt_price(low)}</span>円 '
                f'<span style="color:#26a69a">({low_pct:+.1f}%)</span></div>')
    return (f'<div style="font-size:11px;color:#7c83ff;margin:3px 0 5px">'
            f'📍 押し目 <span style="color:#e0e0f0">{fmt_price(low)}〜{fmt_price(high)}</span>円 '
            f'<span style="color:#26a69a">({low_pct:+.1f}〜{high_pct:+.1f}%)</span></div>')


def make_advice_comment(d: dict) -> str:
    """必ず日本語で回答すること。テクニカル指標からワンポイントアドバイスを生成する。"""
    sigs     = d.get("sigs", {})
    score    = d.get("score", 0)
    rsi      = d.get("rsi",      float("nan"))
    macd_h   = d.get("macd_h",   float("nan"))
    drawdown = d.get("drawdown", float("nan"))

    parts: list[str] = []

    # ── 一目均衡表シグナル評価 ──────────────────────────────────────────────
    if sigs.get("三役好転"):
        parts.append("一目均衡表で三役好転が成立し強気トレンドが継続中")
    elif sigs.get("雲抜け") and sigs.get("上昇トレンド"):
        parts.append("雲上抜けと基準線上維持で上昇基調")
    elif sigs.get("雲抜け"):
        parts.append("雲上抜けを確認したが基準線との位置関係を要確認")
    elif sigs.get("転換GC"):
        parts.append("転換線が基準線をゴールデンクロスし上昇の初動の可能性")
    elif sigs.get("上昇トレンド"):
        parts.append("基準線上で推移しているが雲抜けには至っていない")
    elif score == 0:
        parts.append("現時点で一目均衡表のシグナルは未点灯")

    # ── RSI ────────────────────────────────────────────────────────────────
    if not pd.isna(rsi):
        if rsi >= 75:
            parts.append(f"RSI({rsi:.0f})が過熱圏で短期的な調整リスクに注意")
        elif rsi >= 70:
            parts.append(f"RSI({rsi:.0f})が過熱ゾーン入り口のため利確タイミングを意識")
        elif rsi <= 25:
            parts.append(f"RSI({rsi:.0f})が売られすぎ水準で自律反発に期待")
        elif rsi <= 30:
            parts.append(f"RSI({rsi:.0f})が売られすぎ圏に接近")

    # ── MACD ───────────────────────────────────────────────────────────────
    if not pd.isna(macd_h):
        if macd_h > 0 and not any(parts):
            parts.append("MACDヒストグラムがプラス圏で短期的な上昇モメンタム継続")
        elif macd_h < 0 and score == 0:
            parts.append("MACDヒストグラムがマイナス圏で下降圧力あり")

    # ── 52週高値比 ──────────────────────────────────────────────────────────
    if not pd.isna(drawdown):
        if drawdown >= -3:
            parts.append("52週高値圏で推移しており相対的な強さを維持")
        elif drawdown <= -40:
            parts.append(f"52週高値から{abs(drawdown):.0f}%下落しており底打ち確認が重要")

    # ── 出来高 ─────────────────────────────────────────────────────────────
    if sigs.get("出来高急増"):
        parts.append("出来高急増で市場参加者の関心が高まっている")

    if not parts:
        return ""

    sentence = "。".join(parts[:2]) + "。"
    return (
        f'<div style="font-size:11px;color:#aab8c4;margin-top:6px;padding-top:6px;'
        f'border-top:1px solid #1a1a3a;line-height:1.6">'
        f'💬 {sentence}</div>'
    )


def render_index_card(d: dict) -> str:
    if d.get("error") or d["last"] is None:
        return card(f'<div style="color:#888">{d["name"]} — {d.get("error_msg", "データ取得失敗")}</div>')
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
        + _stats(d) + make_advice_comment(d) + _toggle(d) + render_analyst_section(d.get("analyst", {}))
    )


def render_holding_card(d: dict) -> str:
    if d.get("error") or d["last"] is None:
        return card(f'<div style="color:#888">{d["name"]} — {d.get("error_msg", "データ取得失敗")}</div>')
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
        + _stop_recommend_html(d) + _stats(d) + make_advice_comment(d) + _toggle(d) + warn + render_analyst_section(d.get("analyst", {}))
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
        + _dip_buy_html(d) + _stats(d) + make_advice_comment(d) + _toggle(d) + render_analyst_section(d.get("analyst", {}))
    )


def render_surge_card(d: dict) -> str:
    pc  = _pct_color(d["pct"])
    rsi = d.get("rsi", float("nan"))
    rc  = ("#ef5350" if not pd.isna(rsi) and rsi >= 70
           else "#26a69a" if not pd.isna(rsi) and rsi <= 30 else "#aaa")
    rsi_str = f"{rsi:.1f}" if not pd.isna(rsi) else "—"
    return card(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">'
        f'<div><span style="font-weight:600;font-size:14px">{d["name"]}</span>'
        f' <span style="font-size:11px;color:#666">{d["symbol"]}</span></div>'
        f'<div style="text-align:right">'
        f'<span style="font-size:16px;font-weight:700;color:#e0e0f0">{fmt_price(d["last"])}</span>'
        f' <span style="font-size:13px;color:{pc}">+{d["pct"]:.2f}%</span></div></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#666">'
        f'<span>出来高 <span style="color:#ffd600;font-weight:600">{d["vol_ratio"]:.1f}×</span></span>'
        f'<span>RSI <span style="color:{rc}">{rsi_str}</span></span>'
        f'<span>時価総額 <span style="color:#aaa">{_market_cap_str(d.get("market_cap"))}</span></span>'
        f'</div>'
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    now_jst = datetime.now(JST)
    os.makedirs("docs", exist_ok=True)

    _err_base = lambda sym, nm: {"symbol": sym, "name": nm, "error": True,
                                 "error_msg": "データ取得失敗", "last": None, "pct": 0.0,
                                 "score": 0, "sigs": {}, "last_date_str": "", "analyst": {"news": []}}
    idx_rows = []
    for i in cfg["indices"]:
        try:
            idx_rows.append(build_data(i["symbol"], i["name"]))
        except Exception as e:
            print(f"[WARN] idx {i['symbol']}: {e}", file=sys.stderr)
            idx_rows.append(_err_base(i["symbol"], i["name"]))

    hold_rows = []
    for h in cfg["holdings"]:
        try:
            d = build_data(h["symbol"], h["name"])
            d["cost"]      = h["cost"]
            d["stop_loss"] = h.get("stop_loss")
            d["pnl"]       = (d["last"] / h["cost"] - 1) * 100 if d["last"] else None
            hold_rows.append(d)
        except Exception as e:
            print(f"[WARN] holding {h['symbol']}: {e}", file=sys.stderr)

    max_n     = cfg.get("max_candidates", 10)
    cand_rows = []
    for c in cfg["candidates"]:
        try:
            cand_rows.append(build_data(c["symbol"], c["name"]))
        except Exception as e:
            print(f"[WARN] cand {c['symbol']}: {e}", file=sys.stderr)
    cand_rows = [d for d in cand_rows if not d["error"]]
    max_price = cfg.get("max_price")
    if max_price is not None:
        cand_rows = [d for d in cand_rows if d["last"] is not None and d["last"] <= max_price]
    cand_rows.sort(key=lambda x: (-x["score"], -x["pct"]))
    top = cand_rows[:max_n]

    # 急騰スクリーニング
    surge_syms = fetch_surge_from_kabutan(max_items=60)
    surge_cfg_list = cfg.get("surge_candidates", [])
    if surge_syms:
        name_map = {c["symbol"]: c["name"] for c in surge_cfg_list}
        candidates_to_check = [
            (sym, name_map.get(sym, sym.replace(".T", ""))) for sym in surge_syms
        ]
    else:
        candidates_to_check = [(c["symbol"], c["name"]) for c in surge_cfg_list]
    surge_rows = []
    for sym, nm in candidates_to_check:
        try:
            d = build_surge_data(sym, nm)
            if d is not None:
                surge_rows.append(d)
        except Exception as e:
            print(f"[WARN] surge {sym}: {e}", file=sys.stderr)
    surge_rows.sort(key=lambda x: -x["surge_score"])
    top_surge = surge_rows[:5]

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
  [['d','cd_','bd_'],['m','c5_','b5_'],['1','c1_','b1_']].forEach(function(r){{
    var el=document.getElementById(r[1]+id),btn=document.getElementById(r[2]+id);
    if(el)el.style.display=t===r[0]?'block':'none';
    if(btn)btn.className=t===r[0]?'tab active':'tab';
  }});
}}
</script>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0 10px">
  <div style="font-size:20px;font-weight:800;color:#7c83ff">kabu-watch</div>
  <div style="font-size:10px;color:#555;text-align:right;line-height:1.5">{header_note}</div>
</div>

<h2>保有銘柄</h2>
{"".join(render_holding_card(d) for d in hold_rows)}

<h2>マーケット</h2>
{"".join(render_index_card(d) for d in idx_rows)}

<h2>スクリーナー TOP{len(top)}</h2>
<div style="font-size:11px;color:#555;margin-bottom:8px">スコア = 雲抜け+三役好転+上昇トレンド+転換GC+出来高急増 (最大5)</div>
{"".join(render_candidate_card(d) for d in top) or card('<div style="color:#888;text-align:center">候補銘柄なし</div>')}

<h2>注目急騰5選</h2>
<div style="background:#1a0505;border:1px solid #cc2200;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:#ff5533">⚠ 急騰銘柄は値動きが激しく高リスクです。必ず逆指値を設定してください</div>
<div style="font-size:11px;color:#555;margin-bottom:8px">フィルター: 5,000円以下・当日+3%以上・出来高3倍以上・時価総額500億円以下 | スコア = 出来高倍率×騰落率</div>
{"".join(render_surge_card(d) for d in top_surge) or card('<div style="color:#888;text-align:center">本日の急騰候補なし</div>')}

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
