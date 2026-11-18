"""Accounting baseline and influential-SKU robustness (base setting) -> outputs/influence_check.json
- opening stock of the 368 test SKUs at cost (common to all controllers; not charged in the replay value)
- for each controller: gain over B0 with the five SKUs of largest |gain| removed, and with the five SKUs of
  largest firm value removed; leave-one-out range of the total gain; 95% bootstrap intervals (2,000 draws)."""
import json, os
import numpy as np, pandas as pd
from env2 import load_inputs
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
D = load_inputs(); sk = D["skus"]; te = D["sets"]["test"]
ev = pd.read_csv(os.path.join(OUT, "sim3_eval_base.csv")); P = ev.pivot(index="sku", columns="policy", values="profit")
B0 = "B0 Firm's decisions"
GROUPS = {"QMIX": "qmix_s", "MAPPO": "mappo_s", "MADDPG": "maddpg_s"}
for lab, pre in GROUPS.items():
    P[lab] = P[[c for c in P.columns if c.startswith(pre)]].mean(1)
CTRL = ["B1 Rule-based ladder", "B2 Newsvendor", "B3 Newsvendor, risk-adjusted", "B4 Periodic review (s,S)",
        "B3c Newsvendor + clearance ladder", "B4c (s,S) + clearance ladder", "B5 Multistage predict-then-optimize",
        "QMIX", "MAPPO", "MADDPG", "Hindsight optimum"]
rng = np.random.default_rng(0)
opening = float(sum(sk[k].c * sk[k].q4 for k in te)) / 1e6
res = {"opening_stock_at_cost": opening, "B0_value": float(P[B0].sum() / 1e6), "controllers": {}}


BI_ALL = [rng.integers(0, len(P), len(P)) for _ in range(2000)]   # same draws as analyze3.py (base setting)
rng2 = np.random.default_rng(1)


def summ(g, full=False):
    n = len(g); idx = BI_ALL if full else [rng2.integers(0, n, n) for _ in range(2000)]
    bi = [g[i].sum() for i in idx]
    lo, hi = np.percentile(bi, [2.5, 97.5])
    return {"gain": float(g.sum()), "ci": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


top_firm = P[B0].sort_values(ascending=False).index[:5]
for c in CTRL:
    g = (P[c] - P[B0]) / 1e6
    top_c = g.abs().sort_values(ascending=False).index[:5]
    loo = g.sum() - g.values
    res["controllers"][c] = {"all": summ(g.values, full=True),"drop_top5_abs_gain": summ(g.drop(top_c).values),
                             "drop_top5_firm_value": summ(g.drop(top_firm).values),
                             "loo_min": float(loo.min()), "loo_max": float(loo.max())}
json.dump(res, open(os.path.join(OUT, "influence_check.json"), "w"), indent=1)
print("opening stock at cost %.1f M; B0 value %.1f M" % (opening, res["B0_value"]))
for c, v in res["controllers"].items():
    f = lambda d: "%+.1f [%.1f, %.1f]%s" % (d["gain"], d["ci"][0], d["ci"][1], "*" if d["excludes_zero"] else "")
    print(f"{c:40s} all {f(v['all'])} | -top5|g| {f(v['drop_top5_abs_gain'])} | -top5 firm {f(v['drop_top5_firm_value'])} | LOO {v['loo_min']:.1f}..{v['loo_max']:.1f}")
