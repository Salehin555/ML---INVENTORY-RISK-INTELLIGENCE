"""How concentrated are the losses of early clearance? (base case, t0=3,6,9)"""
import json, re
src = open("decision_value.py", encoding="utf-8").read()
src = src[:src.index("out = {}")]
exec(src)
res = {}
for t0 in [3, 6, 9]:
    F, S, E = build(t0)
    X = F; stocked = ((E.q0 > 0) & (E.c > 0) & (E.flag == 0)).values
    fit_m = E.part.isin(["train", "val"]).values; mt = stocked & (E.part == "test").values
    m = fit_risk(X.values[fit_m], E.dead.values[fit_m]); pr = m.predict_proba(X.values)[:, 1]
    keep, clear = costs(E, 0.5, 0.08); R = 12 - t0
    c, p, q0 = E.c.values, E.p.values, E.q0.values; s_hat = F["SalesQtyCM_mean"].values
    hold = KAPPA * R / 12 * c * q0
    S_ = hold + 0.5 * (1 - np.exp(-0.08 * R)) * c * q0
    L_ = 0.5 * c * q0 + np.maximum(p - c, 0) * s_hat * R - hold
    d = pr * S_ > (1 - pr) * L_
    diff = (np.where(d, clear, keep) - keep)[mt]          # >0 = loss vs keep
    srt = np.sort(diff)[::-1]
    res[t0] = {"total_vs_keep": float(-diff.sum()), "worst1": float(srt[0]), "worst3": float(srt[:3].sum()),
               "total_excl_worst3": float(-(diff.sum() - srt[:3].sum())), "n_cleared_that_sold": int((d[mt] & (E.dead.values[mt] == 0)).sum()),
               "n_cleared": int(d[mt].sum())}
    print(t0, {k: (round(v / 1e6, 2) if isinstance(v, float) else v) for k, v in res[t0].items()})
json.dump(res, open(os.path.join(OUT, "dv_concentration.json"), "w"), indent=1)
