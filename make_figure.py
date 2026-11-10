#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the results table in docs/results.md.

    python make_figure.py

Writes docs/images/decide-light.svg and docs/images/decide-dark.svg, and prints the sentence the figure's alt text uses.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "docs", "results.md")
OUT = os.path.join(HERE, "docs", "images")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "bar": "#2a78d6"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "bar": "#3987e5"},
}
# (row in the results table, label in the figure)
FORMS = [
    ("RTX 5080 Laptop, GenAI CUDA, fp16", "RTX 5080 Laptop: ONNX Runtime GenAI on CUDA, fp16"),
    ("Intel GPU, static graph of 128 tokens, int8 weights", "Intel GPU: static graph of 128 tokens, int8 weights"),
    ("Intel NPU, the same graph in ONNX Runtime + OpenVINO EP", "Intel NPU: static graph of 128 tokens, OpenVINO EP"),
    ("CPU, GenAI OpenVINO EP, int8", "CPU: ONNX Runtime GenAI with the OpenVINO EP, int8"),
]
TITLE = "d1-3B: time to answer one question at 101 input tokens"
NOTE = "Milliseconds on the fastest setup of each processor, one laptop. Shorter is faster."
WIDTH, BAR, VALUE_ROOM = 860, 22, 70


def times():
    """Return [(label, ms at 101 tokens)] for FORMS, read from the results table."""
    if not os.path.exists(SOURCE):
        sys.exit(f"make_figure: {SOURCE} not found")
    with open(SOURCE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for row, label in FORMS:
        line = next((l for l in lines if l.startswith("| " + row + " |")), None)
        if line is None:
            sys.exit(f"make_figure: no row {row!r} in docs/results.md")
        cells = [c.strip() for c in line.strip("|").split("|")]
        parts = [p.strip().rstrip("†").strip() for p in cells[3].split("/")]
        try:
            out.append((label, float(parts[1])))
        except (IndexError, ValueError):
            sys.exit(f"make_figure: no time at 101 tokens in the row {row!r}: {cells[3]!r}")
    return out


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def draw(rows, theme, desc):
    c = THEMES[theme]
    unit = (WIDTH - VALUE_ROOM) / max(v for _, v in rows)
    parts = [text(0, 28, TITLE, 20, c["text"], "600"), text(0, 52, NOTE, 14, c["muted"])]
    y = 88
    for label, value in rows:
        parts.append(text(0, y, label, 15, c["text"]))
        w = value * unit
        parts.append(f'<rect x="0" y="{y + 8:g}" width="{w:g}" height="{BAR}" rx="4" fill="{c["bar"]}"/>')
        parts.append(text(w + 8, y + 8 + 16, f"{value:g} ms", 15, c["text"], "600"))
        y += 62
    height = y - 62 + 8 + BAR + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    rows = times()
    desc = " ".join(f"{label}: {value:g} ms." for label, value in rows)
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"decide-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(draw(rows, theme, desc))
    print(desc)


if __name__ == "__main__":
    main()
