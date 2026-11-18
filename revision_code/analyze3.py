"""Statistics, tables and figures for the final ledger-simulator comparison."""
import json, glob, os
import numpy as np, pandas as pd
from scipy import stats
from common import DATA
from env2 import load_inputs
from figstyle import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
B0 = "B0 Firm's decisions"
GROUPS = {"QMIX": "qmix_s", "MAPPO": "mappo_s", "MADDPG": "maddpg_s",
          "QMIX without structural covariates": "qmix-dynamic_s", "QMIX, two agents (automatic flag)": "qmix-auto_s"}
SETS = ["base", "low lift", "high lift", "low recovery", "high recovery", "no decline", "capped terminal"]
rng = np.random.default_rng(0)

# SKUs whose demand was never censored: recorded closing stock positive in every month 4..11
D = load_inputs(); te = D["sets"]["test"]
cl = pd.read_csv(DATA, usecols=["ProductCode", "Month", "ClosingStkCM"]).pivot(index="ProductCode", columns="Month", values="ClosingStkCM")
uncensored = set(k for k in te if (cl.loc[k, 4:11] > 0).all())

res = {}
for st in SETS:
    df = pd.read_csv(os.path.join(OUT, f"sim3_eval_{st.replace(' ', '_')}.csv"))
    P = df.pivot(index="sku", columns="policy", values="profit")
    n = len(P); BI = [rng.integers(0, n, n) for _ in range(2000)]
    base = P[B0].values; rows = {}
    def summ(v):
        g = v - base
        return {"profit": float(v.sum() / 1e6), "gain": float(g.sum() / 1e6),
                "gain_ci": np.percentile([g[i].sum() / 1e6 for i in BI], [2.5, 97.5]).tolist(),
                "better": float((g > 1).mean()), "worse": float((g < -1).mean())}
    for col in P.columns:
        if any(col.startswith(p) for p in GROUPS.values()): continue
        rows[col] = summ(P[col].values)
        sub = df[df.policy == col]
        rows[col].update({"liquidated": float(sub.liquidated.mean()), "discount_share": float(sub.discount_share.mean()), "stop_share": float(sub.stop_share.mean())})
    for lab, pre in GROUPS.items():
        tags = sorted(c for c in P.columns if c.startswith(pre))
        if not tags: continue
        M = np.array([P[t].values for t in tags]); sm = (M - base).sum(1) / 1e6
        r = summ(M.mean(0)); r.update({"seed_gains": sm.tolist(), "gain_sd": float(sm.std(ddof=1)), "gain_min": float(sm.min()),
                                        "gain_max": float(sm.max()), "seeds_beat_B0": int((sm > 0).sum()), "n": len(tags)})
        sub = df[df.policy.isin(tags)]
        r.update({"liquidated": float(sub.liquidated.mean()), "discount_share": float(sub.discount_share.mean()), "stop_share": float(sub.stop_share.mean())})
        # paired SKU bootstrap: seed-averaged learned policy minus planner B5
        dlt = M.mean(0) - P["B5 Multistage predict-then-optimize"].values
        r["minus_B5"] = float(dlt.sum() / 1e6); r["minus_B5_ci"] = np.percentile([dlt[i].sum() / 1e6 for i in BI], [2.5, 97.5]).tolist()
        rows[lab] = r
    rows["_B0"] = float(base.sum() / 1e6); rows["_n"] = n
    if st == "base":
        unc = [k for k in P.index if k in uncensored]
        rows["_uncensored"] = {"n": len(unc), **{c: float((P.loc[unc, c] - P.loc[unc, B0]).sum() / 1e6) for c in
                                                ["B2 Newsvendor", "B3 Newsvendor, risk-adjusted", "B4 Periodic review (s,S)", "B5, purchasing only",
                                                 "B2 Newsvendor, tuned on validation", "B4 (s,S), tuned on validation"]}}
        sm = {lab: np.array(rows[lab]["seed_gains"]) for lab in GROUPS if lab in rows}
        tests = {}
        for a, b in [("QMIX", "MAPPO"), ("QMIX", "MADDPG"), ("MAPPO", "MADDPG"), ("QMIX", "QMIX without structural covariates"),
                     ("QMIX", "QMIX, two agents (automatic flag)")]:
            tests[f"{a} vs {b}"] = {"diff": float(sm[a].mean() - sm[b].mean()), "p_welch": float(stats.ttest_ind(sm[a], sm[b], equal_var=False).pvalue),
                                    "p_levene": float(stats.levene(sm[a], sm[b]).pvalue)}
        for a in sm: tests[f"{a} vs B0"] = {"p": float(stats.ttest_1samp(sm[a], 0).pvalue)}
        rows["_tests"] = tests
        fl = json.load(open(os.path.join(OUT, "sim3_flags_base.json")))
        rows["_flags"] = {lab: [max(fl[t]) for t in sorted(fl) if t.startswith(pre)] for lab, pre in GROUPS.items()}
    res[st] = rows
json.dump(res, open(os.path.join(OUT, "sim3_summary.json"), "w"), indent=1)

b = res["base"]
for k, v in b.items():
    if k.startswith("_"): continue
    extra = f" sd {v['gain_sd']:.1f} [{v['gain_min']:.1f},{v['gain_max']:.1f}] beat {v['seeds_beat_B0']} | minus B5 {v['minus_B5']:+.1f} {np.round(v['minus_B5_ci'],1)}" if "gain_sd" in v else ""
    print(f"{k:45s} {v['profit']:7.1f} {v['gain']:+7.2f} {np.round(v['gain_ci'],1)} b{v['better']:.2f} w{v['worse']:.2f} liq{v['liquidated']:.2f} disc{v['discount_share']:.2f}{extra}")
print("B0", b["_B0"], "uncensored", b["_uncensored"])
print(json.dumps(b["_tests"], indent=1)); print({k: np.round(v, 2).tolist() for k, v in b["_flags"].items()})
for st in SETS[1:]:
    r = res[st]
    print(st, {k: (round(r[k]["gain"], 1), np.round(r[k]["gain_ci"], 1).tolist()) for k in ["B1 Rule-based ladder", "B5 Multistage predict-then-optimize", "B5, stated parameters", "Hindsight optimum", "QMIX", "MAPPO", "MADDPG"] if k in r})

# figures: drawn from the saved summary
import plot_results  # noqa: E402,F401
