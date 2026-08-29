#!/usr/bin/env python3
"""Pure-stdlib PNG icon generator (no Pillow) for kabu-watch PWA icons.

Draws a simple candlestick-chart glyph (brand purple #7c83ff bars on a
dark #0b0b17 rounded-square background) at whatever size is requested.
"""
import struct
import zlib
import sys


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))


def make_png(path: str, size: int) -> None:
    bg = (11, 11, 23)        # #0b0b17
    purple = (124, 131, 255)  # #7c83ff
    teal = (38, 166, 154)     # #26a69a  (down candle, for variety)
    px = [[bg for _ in range(size)] for _ in range(size)]

    def fill_rect(x0, y0, x1, y1, color):
        for y in range(max(0, y0), min(size, y1)):
            for x in range(max(0, x0), min(size, x1)):
                px[y][x] = color

    # rounded-corner mask: clip the 4 corners of the background square
    corner = max(1, size // 8)
    for y in range(corner):
        for x in range(corner):
            if (x - corner) ** 2 + (y - corner) ** 2 > corner ** 2:
                px[y][x] = None
                px[y][size - 1 - x] = None
                px[size - 1 - y][x] = None
                px[size - 1 - y][size - 1 - x] = None

    # three simple candlesticks of varying height, rising left-to-right
    n = 3
    margin = size * 0.16
    gap = size * 0.08
    bar_w = (size - 2 * margin - (n - 1) * gap) / n
    heights = [0.34, 0.52, 0.74]
    colors = [teal, purple, purple]
    base_y = size * 0.82
    for i, (h_frac, color) in enumerate(zip(heights, colors)):
        x0 = margin + i * (bar_w + gap)
        x1 = x0 + bar_w
        h = size * h_frac
        y0 = base_y - h
        y1 = base_y
        # wick
        wick_x0 = x0 + bar_w * 0.42
        wick_x1 = x0 + bar_w * 0.58
        fill_rect(int(wick_x0), int(y0 - size * 0.06), int(wick_x1), int(y0), color)
        fill_rect(int(x0), int(y0), int(x1), int(y1), color)

    raw = bytearray()
    for y in range(size):
        raw.append(0)  # no filter
        for x in range(size):
            c = px[y][x]
            if c is None:
                raw.extend((11, 11, 23, 0))
            else:
                raw.extend((*c, 255))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    print(f"[OK] wrote {path} ({size}x{size}, {len(png):,} bytes)")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, sz in [("icon-192.png", 192), ("icon-512.png", 512),
                      ("apple-touch-icon.png", 180)]:
        make_png(f"{out_dir}/{name}", sz)
