"""Score run_genai.py output against a reference: same top answer and the largest probability difference.

    python score.py results.jsonl reference_probs.json [--json-out score.json]
"""

import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("reference")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    reference = json.load(open(args.reference, encoding="utf-8"))["text"]
    rows = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    meta = rows[0] if rows and rows[0].get("_meta") else {}
    rows = [row for row in rows if not row.get("_meta")]

    same = total = 0
    worst = (0.0, None)
    diffs, flips, refused, errors, times = [], [], [], [], []
    for row in rows:
        if "refused" in row:
            refused.append((row["q"], row["refused"]))
            continue
        if "error" in row:
            errors.append((row["q"], row["error"][:300]))
            continue
        ref = reference.get(row["q"])
        if ref is None:
            continue
        keys = ref["keys"]
        ref_probs = dict(zip(keys, ref["probs"]))
        probs = row["probs"]
        if set(probs) != set(keys):
            errors.append((row["q"], f"keys differ: {sorted(probs)} vs {sorted(keys)}"))
            continue
        total += 1
        top = max(keys, key=lambda k: probs[k])
        ref_top = max(keys, key=lambda k: ref_probs[k])
        same += top == ref_top
        delta = max(abs(probs[k] - ref_probs[k]) for k in keys)
        diffs.append(delta)
        if delta > worst[0]:
            worst = (delta, row["q"])
        if top != ref_top:
            flips.append({"q": row["q"], "got": top, "ref": ref_top, "near_tie": ref["near_tie"], "delta": delta,
                          "p": {k: round(probs[k], 4) for k in keys}, "ref_p": {k: round(ref_probs[k], 4) for k in keys}})
        times.append((row["tokens"], row["ms"]))

    summary = {
        "results": args.results,
        "meta": meta,
        "questions_scored": total,
        "same_top": same,
        "max_abs_dp": worst[0],
        "max_abs_dp_at": worst[1],
        "median_abs_dp": statistics.median(diffs) if diffs else None,
        "flips": flips,
        "refused": refused,
        "errors": errors,
    }
    if times:
        times.sort()
        def bucket(lo, hi):
            xs = [ms for tok, ms in times if lo <= tok < hi]
            return {"n": len(xs), "median_ms": round(statistics.median(xs), 1)} if xs else None
        summary["ms_by_tokens"] = {"<64": bucket(0, 64), "64-127": bucket(64, 128), "128-255": bucket(128, 256),
                                   "256-511": bucket(256, 512), "512-1023": bucket(512, 1024), "1024+": bucket(1024, 10**9)}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
