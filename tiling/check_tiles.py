"""Compare the public tiling code (hf-to-litertlm d1_3b_work/host/d1_vision.py) with transformers' Lfm2VlImageProcessor, bit for bit.

    python check_tiles.py --d1_vision <dir with d1_vision.py> --model <d1-3B folder> --pictures <dir> [--pictures <dir>] --out tiles.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1_vision", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pictures", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guard", action="store_true", help="flip one bit of one pixel first: every picture must then read as different")
    args = parser.parse_args()
    sys.path.insert(0, args.d1_vision)
    import d1_vision as V
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    for folder in args.pictures:
        with open(os.path.join(folder, "requests.json"), encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            image = Image.open(os.path.join(folder, record["picture"])).convert("RGB")
            capped = V.cap_pixels(image)
            ours = V.preprocess(capped)
            theirs = processor.image_processor(images=[capped], return_tensors="np")
            ref_pixels = np.asarray(theirs["pixel_values"], dtype=np.float32)
            ref_mask = np.asarray(theirs["pixel_attention_mask"])
            ref_shapes = np.asarray(theirs["spatial_shapes"])
            our_pixels = np.stack([t.pixels for t in ours.tiles])
            our_mask = np.stack([t.mask for t in ours.tiles])
            our_shapes = np.asarray([t.grid for t in ours.tiles])
            if args.guard:
                our_pixels.view(np.uint32).flat[our_pixels.size // 2] ^= 1
            same_pixels = our_pixels.shape == ref_pixels.shape and np.array_equal(our_pixels.view(np.uint32), ref_pixels.view(np.uint32))
            row = {
                "id": record["id"], "size": image.size, "capped": capped.size, "tiles": len(ours.tiles), "rows": ours.rows, "cols": ours.cols,
                "same_pixels_bits": bool(same_pixels), "same_mask": bool(np.array_equal(our_mask, ref_mask)),
                "same_shapes": bool(np.array_equal(our_shapes, ref_shapes)), "shapes": [list(our_pixels.shape), list(ref_pixels.shape)],
            }
            if not same_pixels and our_pixels.shape == ref_pixels.shape:
                row["max_abs_diff"] = float(np.abs(our_pixels - ref_pixels).max())
                row["bits_differ"] = int((our_pixels.view(np.uint32) != ref_pixels.view(np.uint32)).sum())
            rows.append(row)
            print(row, flush=True)
    summary = {"pictures": len(rows), "all_bits_equal": sum(r["same_pixels_bits"] for r in rows), "all_mask_equal": sum(r["same_mask"] for r in rows),
               "all_shapes_equal": sum(r["same_shapes"] for r in rows), "tiles": sorted({r["tiles"] for r in rows})}
    print(json.dumps(summary))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=0)


if __name__ == "__main__":
    main()
