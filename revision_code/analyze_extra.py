"""Loss decomposition and flag informativeness (base setting) -> outputs/extra_analysis.json
Uses outputs/sim3_actions_base.pkl and sim3_eval_base.csv written by evaluate3.py base."""
import json, os, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from env2 import load_inputs
from sim2 import T0, T1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
D = load_inputs(); sk = D["skus"]; te = D["sets"]["test"]; PH = D["phat"]
LOG = pickle.load(open(os.path.join(OUT, "sim3_actions_base.pkl"), "rb"))
ev = pd.read_csv(os.path.join(OUT, "sim3_eval_base.csv")); P = ev.pivot(index="sku", columns="policy", values="profit")
B0 = "B0 Firm's decisions"
GROUPS = {"QMIX": "qmix_s", "MAPPO": "mappo_s", "MADDPG": "maddpg_s",
          "QMIX without structural covariates": "qmix-dynamic_s", "QMIX, two agents (automatic flag)": "qmix-auto_s"}
SINGLE = ["B1 Rule-based ladder", "B3c Newsvendor + clearance ladder", "B4c (s,S) + clearance ladder",
          "B5 Multistage predict-then-optimize", "B5, firm's orders kept", "B5, purchasing only"]


def eff(A):
    L = A[:, 2].copy(); m = A[:, 0]
    L[((L == 1) | (L == 2)) & (m < 2)] = 0; L[(L == 3) & (m < 3)] = 0; L[A[:, 3] == 1] = 0
    return L


def split(tag):
    """gain over B0 (million BDT) by item type: not liquidated / liquidated and later sold / liquidated, never sold"""
    out = {"kept": [0, 0.0], "liq_sold_later": [0, 0.0], "liq_no_later_sale": [0, 0.0], "liq_low_stock_firm_reordered": [0, 0.0]}
    for k in te:
        g = (P.loc[k, tag] - P.loc[k, B0]) / 1e6
        L = eff(LOG[(tag, k)]); idx = np.where(L == 3)[0]
        if not len(idx):
            out["kept"][0] += 1; out["kept"][1] += g; continue
        t = T0 + idx[0]                                      # liquidation month
        later = sk[k].d[t:T1].sum() > 0                      # recorded sales after month t, to November
        key = "liq_sold_later" if later else "liq_no_later_sale"
        out[key][0] += 1; out[key][1] += g
        if later and sk[k].g[t:T1].sum() > 0 and sk[k].q4 <= 1:
            out["liq_low_stock_firm_reordered"][0] += 1; out["liq_low_stock_firm_reordered"][1] += g
    return out


res = {"split": {}, "flags": {}}
for c in SINGLE:
    res["split"][c] = split(c)
for lab, pre in GROUPS.items():
    tags = sorted({t for (t, _) in LOG if isinstance(t, str) and t.startswith(pre)})
    parts = [split(t) for t in tags]
    res["split"][lab] = {k: [float(np.mean([p[k][0] for p in parts])), float(np.mean([p[k][1] for p in parts]))] for k in parts[0]}
    # flag informativeness: item-months before exit; outcome = no recorded sale from month t to November
    aucs, aucs_p, blocked, collapsed = [], [], [], 0
    for t_ in tags:
        key = "dynamic" if "-dynamic" in t_ else "both"
        f, y, ph, ch, ef = [], [], [], 0, 0
        for k in te:
            A = LOG[(t_, k)]; L = eff(A)
            for j in range(A.shape[0]):
                if A[j, 3] == 1: continue
                tt = T0 + j
                f.append(min(A[j, 0], 2) if A[j, 0] >= 2 else 0)        # none and low merged (same economics)
                y.append(float(sk[k].d[tt - 1:T1].sum() == 0)); ph.append(PH[key][k][j])
                if A[j, 2] > 0: ch += 1; ef += L[j] > 0
        f, y = np.array(f), np.array(y)
        aucs.append(roc_auc_score(y, f) if len(set(f)) > 1 else 0.5); aucs_p.append(roc_auc_score(y, ph))
        blocked.append(1 - ef / ch if ch else 0.0)
        collapsed += int(np.bincount(f, minlength=3).max() / len(f) > 0.95)
    res["flags"][lab] = {"auc_flag_mean": float(np.mean(aucs)), "auc_flag_min": float(np.min(aucs)), "auc_flag_max": float(np.max(aucs)),
                         "auc_phat_mean": float(np.mean(aucs_p)), "blocked_share_mean": float(np.mean(blocked)),
                         "blocked_min": float(np.min(blocked)), "blocked_max": float(np.max(blocked)), "seeds_one_flag_95": collapsed}
# reference: rule-based ladder flags
f, y = [], []
for k in te:
    A = LOG[("B1 Rule-based ladder", k)]
    for j in range(A.shape[0]):
        if A[j, 3] == 1: continue
        f.append(A[j, 0] if A[j, 0] >= 2 else 0); y.append(float(sk[k].d[T0 + j - 1:T1].sum() == 0))
res["flags"]["B1 Rule-based ladder"] = {"auc_flag_mean": float(roc_auc_score(y, f))}
json.dump(res, open(os.path.join(OUT, "extra_analysis.json"), "w"), indent=1)
for k, v in res["split"].items():
    print(f"{k:42s}", {a: (round(b[0], 1), round(b[1], 1)) for a, b in v.items()})
for k, v in res["flags"].items():
    print(k, {a: round(b, 3) for a, b in v.items()})
