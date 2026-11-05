"""Median, min and max milliseconds per prompt length from run_genai.py output on speed_requests.json."""
import json
import statistics
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    meta = rows[0]
    by = {}
    for row in rows[1:]:
        if "ms" in row:
            by.setdefault(row["tokens"], []).append(row["ms"])
        else:
            print(path, "row without ms:", str(row)[:200])
    line = " | ".join(f"{t} tok {statistics.median(v):.1f} ms ({min(v):.1f}-{max(v):.1f}, n={len(v)})" for t, v in sorted(by.items()))
    print(f"{path}: ep={meta.get('ep')} load={meta.get('load_s', 0):.1f}s :: {line}")
