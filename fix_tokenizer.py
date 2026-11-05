"""The tokenizer fix from the earlier d1 tutorial (commit 7f090db, step 3): needed for the PyPI 0.19.0 wheels, built from 331d6ebb.
Keeps the unpatched file as tokenizer.json.orig so the failure can be reproduced."""
import json
import shutil
import sys
from pathlib import Path

export = Path(sys.argv[1])
path = export / "tokenizer.json"
orig = export / "tokenizer.json.orig"
if not orig.exists():
    shutil.copy2(path, orig)
tokenizer = json.loads(orig.read_text(encoding="utf-8"))
old = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"
new = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
split = tokenizer["pre_tokenizer"]["pretokenizers"][0]
assert split["pattern"]["Regex"] == old
split["pattern"]["Regex"] = new
path.write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
print("patched", path)
