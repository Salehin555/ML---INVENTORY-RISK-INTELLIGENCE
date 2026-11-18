"""Exact hindsight optimum for one SKU: forward dynamic programming over months 4..11
with the recorded future known.  Every (purchase, clearance) pair is allowed with the
lowest flag that permits it (higher flags only add review cost).  States are stock levels;
equal stock levels are merged keeping the best profit so far.  Gives an upper bound on the
profit of any controller that uses the same action set."""
import numpy as np
from sim2 import step, terminal, T0, T1, PARAMS, min_flag

PAIRS = [(P, L) for P in range(4) for L in range(4)]


def hindsight_value(sku, prm=PARAMS, max_states=200_000):
    states = {round(sku.q4, 9): 0.0}          # stock -> best cumulative profit
    best_exit = -np.inf                       # best value among paths that liquidated
    for t in range(T0, T1 + 1):
        nxt = {}
        for q, v in states.items():
            for P, L in PAIRS:
                r, q1, ex = step(sku, t, q, (min_flag(L), P, L), prm, False)
                if ex:
                    best_exit = max(best_exit, v + r)
                    continue
                k = round(q1, 9)
                if v + r > nxt.get(k, -np.inf):
                    nxt[k] = v + r
        if len(nxt) > max_states:             # merge nearby stock levels (never triggered in our data)
            merged = {}
            for q, v in nxt.items():
                k = round(q, 1)
                if v > merged.get(k, -np.inf): merged[k] = v
            nxt = merged
        states = nxt
    best_keep = max(v + terminal(sku, q, False, prm) for q, v in states.items())
    return max(best_keep, best_exit)
