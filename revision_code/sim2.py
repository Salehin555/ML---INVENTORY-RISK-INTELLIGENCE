"""Ledger-based inventory simulator in money units (BDT).

Each SKU replays its own recorded months.  Months 1-3 are history; decisions are
taken at the start of months 4..11 (eight decisions); the December extract
records stock but no transactions.  Demand in month t is the
recorded sales quantity (a lower bound on true demand when the item ran out).
Stock follows the inventory balance  q_{t+1} = q_t + order_t - sales_t.

Decisions per month
  monitoring  m in {0 none, 1 low, 2 medium, 3 high}
  purchasing  P in {0 stop, 1 small reorder, 2 large reorder, 3 hold}
  clearance   L in {0 hold, 1 discount 10%, 2 discount 20%, 3 liquidate}
Escalation rule: a discount takes effect only if m >= 2 and liquidation only if
m == 3; otherwise the clearance action is ignored.
  stop  : no order;  hold: the order the firm actually placed (recorded receipts)
  small : mean monthly demand observed so far;  large: three times that
  discount d: price (1-d)p, demand multiplied by the lift LIFT[d]
  liquidate : all stock sold at rho*c*exp(-beta*(t-1)); the item exits for good
Monthly profit = revenue - purchase cost - carrying cost (kappa/12 * c * end stock).
At the end of month 11 remaining stock is valued at cost if the item sold in
months 9-11, and at the decayed recovery value otherwise (optionally, only stock up
to `cover` months of September-November sales is valued at cost).
"""
import numpy as np, pandas as pd, os, itertools

T0, T1 = 4, 11                       # first and last decision month (December has no transactions)
KAPPA = 0.20
PARAMS = {"lift10": 1.5, "lift20": 2.0, "rho": 0.5, "beta": 0.08}
REVIEW_COST = 500.0                  # BDT per item-month flagged medium or high (planner review time)
ACTS = list(itertools.product(range(4), range(4)))   # (purchase, clearance)


def min_flag(L):
    return 3 if L == 3 else (2 if L in (1, 2) else 0)


class SKU:
    """Recorded trace of one SKU."""
    def __init__(self, d, g, q_open, c, p, adj=None):
        self.d = d.astype(float); self.g = g.astype(float)          # months 1..12 (index 0..11)
        self.adj = np.zeros(12) if adj is None else adj.astype(float)  # recorded stock adjustments (exogenous)
        self.q4 = float(q_open[T0 - 1]); self.c = float(c); self.p = float(p)
        self.alive_end = self.d[8:11].sum() > 0          # sold in September-November


def order_qty(sku, t, P):
    past = sku.d[:t - 1]
    mu = past.mean() if len(past) else 0.0
    return [0.0, np.ceil(mu), np.ceil(3 * mu), sku.g[t - 1]][P]


def step(sku, t, q, a, prm=PARAMS, exited=False, demand=None):
    """One month.  a = (m, P, L).  Returns profit, new stock, exited."""
    if exited:
        return 0.0, 0.0, True
    m, P, L = a
    review = REVIEW_COST if m >= 2 else 0.0
    if L in (1, 2) and m < 2: L = 0
    if L == 3 and m < 3: L = 0
    o = order_qty(sku, t, P)
    avail = max(q + o + sku.adj[t - 1], 0.0)
    cost = sku.c * o
    if L == 3:
        rev = prm["rho"] * sku.c * np.exp(-prm["beta"] * (t - 1)) * avail
        return rev - cost - review, 0.0, True
    f, lift = [(1.0, 1.0), (0.9, prm["lift10"]), (0.8, prm["lift20"])][L]
    dem = (sku.d[t - 1] if demand is None else demand) * lift
    sales = min(avail, dem)
    q1 = avail - sales
    profit = sales * sku.p * f - cost - KAPPA / 12 * sku.c * q1 - review
    return profit, q1, False


def terminal(sku, q, exited, prm=PARAMS, alive=None):
    if exited or q <= 0:
        return 0.0
    alive = sku.alive_end if alive is None else alive
    low = prm["rho"] * sku.c * np.exp(-prm["beta"] * T1)
    if not alive:
        return low * q
    if prm.get("cover"):                 # only `cover` months of September-November sales valued at cost
        k = min(q, prm["cover"] * sku.d[8:11].mean())
        return sku.c * k + low * (q - k)
    return sku.c * q


def features(sku, t, q, phat, exited):
    """State available at the start of month t (uses months < t only)."""
    past = sku.d[:t - 1]
    mu = past.mean(); rec = past[-3:].mean()
    sold = np.where(past > 0)[0]
    msls = (t - 1 - (sold[-1] + 1)) if len(sold) else t - 1
    margin = (sku.p - sku.c) / sku.p if sku.p > 0 else 0.0
    return np.array([
        np.log1p(q) / 10, min(q / (mu + 0.1), 24) / 24, min(rec / (mu + 0.1), 3) / 3,
        msls / 12, float(len(sold) > 0), np.clip(margin, -1, 1), np.log1p(sku.c) / 15,
        (t - T0) / (T1 - T0), phat, float(exited)], dtype=np.float32)


SD2 = 10


def rollout(sku, policy, phat_row, prm=PARAMS):
    """Run months 4..11 under policy(state, sku, t, q) -> (m, P, L)."""
    q, ex, tot = sku.q4, False, 0.0
    acts = []
    for t in range(T0, T1 + 1):
        s = features(sku, t, q, phat_row[t - T0], ex)
        a = policy(s, sku, t, q, phat_row[t - T0], ex)
        acts.append(tuple(a) + (int(ex),))
        r, q, ex = step(sku, t, q, a, prm, ex)
        tot += r
    tot += terminal(sku, q, ex, prm)
    return tot, acts
