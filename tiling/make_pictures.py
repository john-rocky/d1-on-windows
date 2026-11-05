"""Make a small, reproducible picture set for the image tiling check: invoices with a fine-print total.

Every picture is above the size where d1-3B and transformers split a picture into tiles
(round32(h) * round32(w) > 256 * 32 * 32 * 2.0 = 524,288 pixels). The question asks for the total printed on the page, so
the answer depends on how much of the page's detail reaches the vision model.

    python make_pictures.py --out pictures                                   # 18 pictures, print of 18 to 32 pixels
    python make_pictures.py --out fine --sizes 1600x1200 --fonts 8,10,12,14 --per 5 --seed 2000   # 20 pictures, finer print
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

ITEMS = ["Paper", "Toner", "Stapler", "Binder", "Marker", "Folder", "Envelope", "Label", "Tape", "Pen", "Clip", "Notebook"]


def money(cents):
    return f"${cents // 100}.{cents % 100:02d}"


def picture(seed, size, font_px):
    rng = random.Random(seed)
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_px)
    title = ImageFont.load_default(size=font_px + 6)
    margin = width // 10
    y = height // 12
    draw.text((margin, y), f"INVOICE {rng.randint(10000, 99999)}", fill="black", font=title)
    y += int(font_px * 2.4)
    rows = rng.sample(ITEMS, 8)
    total = 0
    for name in rows:
        cents = rng.randint(120, 9800)
        total += cents
        draw.text((margin, y), name, fill="black", font=font)
        draw.text((width - margin - font_px * 6, y), money(cents), fill="black", font=font)
        y += int(font_px * 1.6)
    y += font_px
    draw.line((margin, y, width - margin, y), fill="black", width=2)
    y += font_px
    draw.text((margin, y), "TOTAL DUE", fill="black", font=font)
    draw.text((width - margin - font_px * 6, y), money(total), fill="black", font=font)
    # Three wrong totals, close to the right one, in a shuffled order.
    wrong = set()
    while len(wrong) < 3:
        other = total + rng.randint(-2500, 2500)
        if other > 0 and other != total:
            wrong.add(other)
    options = [money(total)] + [money(c) for c in wrong]
    rng.shuffle(options)
    return image, money(total), options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="1600x1200,1280x960,1920x1080")
    parser.add_argument("--fonts", default="18,24,32")
    parser.add_argument("--per", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    sizes = [tuple(int(v) for v in item.split("x")) for item in args.sizes.split(",")]
    fonts = [int(v) for v in args.fonts.split(",")]
    os.makedirs(args.out, exist_ok=True)
    records = []
    seed = args.seed
    for size in sizes:
        for font_px in fonts:
            for _ in range(args.per):
                seed += 1
                image, gold, options = picture(seed, size, font_px)
                name = f"invoice_{size[0]}x{size[1]}_f{font_px}_{seed}.png"
                image.save(os.path.join(args.out, name))
                criteria = {f"option_{i}": text for i, text in enumerate(options)}
                gold_key = next(k for k, v in criteria.items() if v == gold)
                records.append({
                    "id": name[:-4],
                    "picture": name,
                    "wh": list(size),
                    "font_px": font_px,
                    "gold": gold_key,
                    "request": {"state": None, "questions": {"total": {
                        "type": "choice",
                        "instructions": "What is the TOTAL DUE printed on this invoice?",
                        "criteria": criteria,
                    }}},
                })
    with open(os.path.join(args.out, "requests.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    print(len(records), "pictures in", args.out)


if __name__ == "__main__":
    main()
