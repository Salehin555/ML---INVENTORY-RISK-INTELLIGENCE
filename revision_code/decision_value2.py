"""Robustness of the decision-value experiment (base case rho=0.5, beta=0.08, target: no further sale).

1. Rule specification: (a) plug-in expected-cost rule (models trained on training+validation SKUs),
   (b) the same rule with probabilities calibrated by isotonic regression on validation SKUs
       (models trained on training SKUs only), (c) a single threshold on the probability,
       chosen on validation SKUs to minimise cost (the specification used in an earlier run).
2. Outcome horizon: 'no further sale' measured to November 2024, or extended with the firm's
   2025 sales (an item that sells in 2025 is not dead; its stock is not written down).
3. Like-for-like comparison: SHAP share of structural covariates in the same model
   (month t0, target 'no further sale', both families), Shapley split of the saving between
   the two families, and the coefficient of prescriptiveness P = (C_keep - C_rule)/(C_keep - C_oracle).
"""
import json
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
src = open(__file__.replace("decision_value2.py", "decision_value.py"), encoding="utf-8").read()
exec(src[:src.index("out = {}")])

# 2025 sales per SKU (extends the outcome beyond the censored 2024 horizon)
x = pd.ExcelFile(os.path.join(ROOT, "data 2025", "Product Cost_Status_Sales Value Database._Year 2025 (1).xlsx"))
s25 = pd.concat([pd.read_excel(x, sh, usecols=["ProductCode", "SalesQtyCM"]) for sh in x.sheet_names])
s25["SalesQtyCM"] = pd.to_numeric(s25.SalesQtyCM, errors="coerce").fillna(0)
sold25 = s25.groupby("ProductCode").SalesQtyCM.sum()

RHO, BETA = 0.5, 0.08
res = {}
for t0 in [3, 6, 9]:
    F, S, E = build(t0)
    R = 12 - t0
    stocked = ((E.q0 > 0) & (E.c > 0) & (E.flag == 0)).values
    mt = stocked & (E.part == "test").values
    for horizon in ["2024", "2024+2025"]:
        dead = E.dead.values.copy()
        if horizon == "2024+2025":
            s = sold25.reindex(E.index)
            dead = np.where(s.notna() & (s.fillna(0) > 0), 0, dead)
        keep = (E.hold - E.marg).values + dead * (1 - RHO * np.exp(-BETA * R)) * (E.c * E.q_end).values
        clear = ((1 - RHO) * E.c * E.q0).values
        c, p, q0 = E.c.values, E.p.values, E.q0.values
        s_hat = F["SalesQtyCM_mean"].values
        hold_est = KAPPA * R / 12 * c * q0
        save_dead = hold_est + RHO * (1 - np.exp(-BETA * R)) * c * q0
        loss_alive = (1 - RHO) * c * q0 + np.maximum(p - c, 0) * s_hat * R - hold_est
        X_all = {"dynamic": F, "structural": S, "both": F.join(S)}
        trm = (E.part == "train").values; vam = (E.part == "val").values; tvm = trm | vam
        out = {"keep": keep[mt].sum(), "oracle": np.minimum(keep, clear)[mt].sum()}
        for name, X in X_all.items():
            m_tv = fit_risk(X.values[tvm], dead[tvm]); p_tv = m_tv.predict_proba(X.values)[:, 1]
            m_tr = fit_risk(X.values[trm], dead[trm]); p_tr = m_tr.predict_proba(X.values)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr[vam], dead[vam]); p_cal = iso.predict(p_tr)
            vm = stocked & vam
            grid = np.linspace(0, 1, 101)
            th = grid[int(np.argmin([np.where(p_tr[vm] > g, clear[vm], keep[vm]).sum() for g in grid]))]
            for spec, pr, rule in [("plug-in", p_tv, None), ("calibrated plug-in", p_cal, None), ("tuned threshold", p_tr, th)]:
                d_ = (pr > rule) if rule is not None else (pr * save_dead > (1 - pr) * loss_alive)
                out[f"{spec}|{name}"] = np.where(d_, clear, keep)[mt].sum()
            if name == "both" and horizon == "2024":
                C = m_tv.get_booster().predict(xgb.DMatrix(X.values[tvm]), pred_contribs=True)[:, :-1]
                a = np.abs(C).mean(0)
                out["shap_structural_share_same_model"] = float(a[[list(X.columns).index(k) for k in STRUCT]].sum() / a.sum())
        for spec in ["plug-in", "calibrated plug-in", "tuned threshold"]:
            v = {n: out["keep"] - out[f"{spec}|{n}"] for n in X_all}          # saving vs keep-all (v(empty)=0)
            sh_s = 0.5 * (v["structural"] + (v["both"] - v["dynamic"]))
            sh_d = 0.5 * (v["dynamic"] + (v["both"] - v["structural"]))
            gap = out["keep"] - out["oracle"]
            out[f"{spec}|gain_struct_over_dyn"] = v["both"] - v["dynamic"]
            out[f"{spec}|shapley_struct"] = sh_s; out[f"{spec}|shapley_dyn"] = sh_d
            for n in X_all: out[f"{spec}|P_{n}"] = v[n] / gap
        res[f"t0={t0}|horizon={horizon}"] = {k: float(v) for k, v in out.items()}
        r = res[f"t0={t0}|horizon={horizon}"]
        print(t0, horizon, "oracle saving %.2f" % ((r["keep"] - r["oracle"]) / 1e6),
              {spec: "dyn %.2f both %.2f gain %.2f P_both %.2f" % ((r["keep"] - r[f"{spec}|dynamic"]) / 1e6, (r["keep"] - r[f"{spec}|both"]) / 1e6,
                                                                r[f"{spec}|gain_struct_over_dyn"] / 1e6, r[f"{spec}|P_both"])
               for spec in ["plug-in", "calibrated plug-in", "tuned threshold"]},
              "shap" if "shap_structural_share_same_model" in r else "", round(r.get("shap_structural_share_same_model", 0), 3), flush=True)
json.dump(res, open(os.path.join(OUT, "decision_value_robust.json"), "w"), indent=1)
