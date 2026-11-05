"""The read-out when a graph from export_ir.py --head hidden returns the hidden state instead of the logits.

Upstream's probabilities() takes a softmax over the whole vocabulary, then a softmax over the options' scores. The first
one's denominator is common to every option and cancels, so only the options' rows of the head matrix are needed.
"""

import numpy as np

SMALLEST_VOCABULARY = 10000  # a hidden state is smaller than this, so read as logits it would answer wrongly without an error


def load_head(path):
    return np.load(path, mmap_mode="r") if path else None


def read_out(up, output, group, head):
    """The option probabilities, from logits (head is None) or from a hidden state and the head matrix."""
    if head is None:
        if output.size < SMALLEST_VOCABULARY:
            raise SystemExit(f"this graph returns {output.size} values, a hidden state: pass --head lm_head.npy")
        return up.probabilities(output, group)
    ids = sorted({token for tokens in group for token in tokens})
    where = {token: n for n, token in enumerate(ids)}
    return up.probabilities(np.asarray(head[ids]) @ output, [[where[token] for token in tokens] for tokens in group])
