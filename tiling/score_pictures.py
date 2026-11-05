"""Compare the two arms of run_pictures.py: right answers against the printed total, same top answer, largest difference.

    python score_pictures.py pictures/requests.json hf.jsonl genai.jsonl [more.jsonl ...]
"""

import json
import sys


def load(path):
    return {row["id"]: row for row in map(json.loads, open(path, encoding="utf-8"))}


def main():
    records = json.load(open(sys.argv[1], encoding="utf-8"))
    reference = load(sys.argv[2])
    arms = {sys.argv[2]: reference}
    for path in sys.argv[3:]:
        arms[path] = load(path)
    for name, rows in arms.items():
        right = sum(rows[r["id"]]["top"] == r["gold"] for r in records if r["id"] in rows)
        n = sum(r["id"] in rows for r in records)
        print(f"{name}: right {right}/{n}")
        for font in sorted({r["font_px"] for r in records}):
            subset = [r for r in records if r["font_px"] == font and r["id"] in rows]
            print(f"  font {font}px: {sum(rows[r['id']]['top'] == r['gold'] for r in subset)}/{len(subset)}")
        if rows is not reference:
            shared = [r for r in records if r["id"] in rows and r["id"] in reference]
            same = sum(rows[r["id"]]["top"] == reference[r["id"]]["top"] for r in shared)
            worst = max(max(abs(rows[r["id"]]["probs"][k] - reference[r["id"]]["probs"][k]) for k in rows[r["id"]]["probs"])
                        for r in shared)
            print(f"  same top as {sys.argv[2]}: {same}/{len(shared)}, largest probability difference {worst:.4f}")
            tokens = sorted(rows[r["id"]]["tokens"] for r in shared)
            print(f"  prompt tokens {tokens[0]}..{tokens[-1]}; reference {min(reference[r['id']]['tokens'] for r in shared)}.."
                  f"{max(reference[r['id']]['tokens'] for r in shared)}")


if __name__ == "__main__":
    main()
