"""Decision value of side information, measured on realised outcomes (no simulator).

At the end of month t0 each stocked SKU is either kept (status quo) or cleared
(stock sold at recovery rate rho of cost, no further purchases).  The cost of
keeping is computed from what actually happened in months t0+1..12: carrying
cost on the recorded stock, minus the gross margin on recorded sales, plus the
write-down of year-end stock if the item made no sale after t0.  The decision
rule clears an item when its estimated risk exceeds a threshold chosen on the
validation SKUs.  Risk models are trained on information available at t0 only,
with three information sets: dynamic, structural, both.
usage: python decision_value.py"""
import json
import xgboost as xgb
from common import *

KAPPA = 0.20
STRUCT = ["PGName_Enc", "MAGName_Enc", "ASGName_Enc", "AGName_Enc", "BrandDesc_Enc"]
# month-specific ERP fields only (fields recorded as of the extraction date are excluded)
DYN_RAW = ["SalesQtyCM", "SalesValCM", "GRDCM", "GRDVALCM", "AddStockCM", "DeductStockCM",
           "OpeningStk", "ClosingStkCM", "ClosingValCM", "CostPrice", "RSP", "GrossMarginPct", "PriceFlag_Any"]

df = pd.read_csv(DATA).sort_values([SKU, MONTH])
tr, va, te = sku_split(df)
part = {**{k: "train" for k in tr}, **{k: "val" for k in va}, **{k: "test" for k in te}}


def build(t0):
    pre = df[df[MONTH] <= t0]; post = df[df[MONTH] > t0]
    g = pre.groupby(SKU)
    F = g[DYN_RAW].agg(["mean", "max", "std"]); F.columns = ["_".join(c) for c in F.columns]
    last = g.last()
    for c in DYN_RAW: F[c + "_last"] = last[c]
    sold = pre[pre.SalesQtyCM > 0].groupby(SKU)[MONTH].max()
    F["months_since_sale_t0"] = t0 - sold.reindex(F.index).fillna(0)
    F["zero_sale_months_t0"] = g.SalesQtyCM.apply(lambda x: (x == 0).sum())
    F["sales_slope_t0"] = g[[MONTH, "SalesQtyCM"]].apply(lambda x: np.polyfit(x[MONTH], x.SalesQtyCM, 1)[0] if len(x) > 1 else 0.0)
    F = F.fillna(0)
    S = last[STRUCT]
    # realised economics after t0
    c = last.CostPrice; p = last.RSP; q0 = last.ClosingStkCM
    hold = post.assign(h=post.ClosingStkCM * post.CostPrice * KAPPA / 12).groupby(SKU).h.sum()
    marg = post.assign(m=post.SalesQtyCM * (post.RSP - post.CostPrice)).groupby(SKU).m.sum()
    sales_after = post.groupby(SKU).SalesQtyCM.sum()
    q_end = post.groupby(SKU).ClosingStkCM.last()
    E = pd.DataFrame({"c": c, "p": p, "q0": q0, "hold": hold, "marg": marg,
                      "sales_after": sales_after, "q_end": q_end,
                      "label": last[TARGET_NUM], "flag": last.PriceFlag_Any}).reindex(F.index).fillna(0)
    E["dead"] = (E.sales_after == 0).astype(int)
    E["part"] = [part[k] for k in E.index]
    E["R"] = 12 - t0
    return F, S, E


def costs(E, rho, delta):
    """Recovery rate rho of cost if cleared now; rho*exp(-delta*R) if the stock of a
    dead item is cleared at year end instead."""
    keep = E.hold - E.marg + E.dead * (1 - rho * np.exp(-delta * E.R)) * E.c * E.q_end
    clear = (1 - rho) * E.c * E.q0
    return keep.values, clear.values


def fit_risk(X, y):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=8, eval_metric="mlogloss")
    m.fit(X, y); return m


out = {}
for t0 in [3, 6, 9]:
    F, S, E = build(t0)
    X_all = {"dynamic": F, "structural": S, "both": F.join(S)}
    stocked = ((E.q0 > 0) & (E.c > 0) & (E.flag == 0)).values
    fit_m = E.part.isin(["train", "val"]).values          # no tuning: train on all non-test SKUs
    mt = stocked & (E.part == "test").values
    s_hat = F["SalesQtyCM_mean"].values                    # mean monthly sales up to t0
    R = 12 - t0
    for target in ["dead", "class"]:
        y = E.dead.values if target == "dead" else E.label.values
        risk = {}
        for name, X in X_all.items():
            m = fit_risk(X.values[fit_m], y[fit_m]); P = m.predict_proba(X.values)
            risk[name] = P[:, 1] if target == "dead" else P[:, 1] + P[:, 2]
            if target == "class" and name == "both" and t0 == 6:
                C = m.get_booster().predict(xgb.DMatrix(X.values[fit_m][:3000]), pred_contribs=True)
                a = np.abs(C[:, :, :-1]).mean(axis=(0, 1))
                out["shap_structural_share_t0_6"] = float(a[[list(X.columns).index(c) for c in STRUCT]].sum() / a.sum())
        for rho, delta in [(0.5, 0.08), (0.3, 0.08), (0.7, 0.08), (0.5, 0.0)]:
            keep, clear = costs(E, rho, delta)
            c, p, q0 = E.c.values, E.p.values, E.q0.values
            hold_est = KAPPA * R / 12 * c * q0
            save_dead = hold_est + rho * (1 - np.exp(-delta * R)) * c * q0          # known at t0
            loss_alive = (1 - rho) * c * q0 + np.maximum(p - c, 0) * s_hat * R - hold_est  # estimated at t0
            r = {"n_test": int(mt.sum()), "stock_value_test": float((c * q0)[mt].sum()),
                 "keep_all": float(keep[mt].sum()), "clear_all": float(clear[mt].sum()),
                 "oracle": float(np.minimum(keep, clear)[mt].sum()),
                 "rule_no_sale_3m": float(np.where(F["months_since_sale_t0"].values >= 3, clear, keep)[mt].sum())}
            dec = {}
            for name in X_all:
                pr = risk[name]
                d_ = pr * save_dead > (1 - pr) * loss_alive
                dec[name] = np.where(d_, clear, keep)[mt]
                r[name] = float(dec[name].sum()); r[name + "_n_cleared"] = int(d_[mt].sum())
                r[name + "_saving_vs_keep"] = r["keep_all"] - r[name]
            rng = np.random.default_rng(0); n = int(mt.sum()); B = [rng.integers(0, n, n) for _ in range(2000)]
            kp = keep[mt]
            for name in X_all:
                r[name + "_saving_ci"] = np.percentile([kp[i].sum() - dec[name][i].sum() for i in B], [2.5, 97.5]).tolist()
            r["both_minus_dynamic"] = r["dynamic"] - r["both"]
            r["both_minus_dynamic_ci"] = np.percentile([dec["dynamic"][i].sum() - dec["both"][i].sum() for i in B], [2.5, 97.5]).tolist()
            key = f"t0={t0}|target={target}|rho={rho}|delta={delta}"
            out[key] = r
            print(key, "stock %.0f | oracle saves %.2f | rule3m %.2f | dyn %.2f %s | str %.2f %s | both %.2f %s | both-dyn %.2f %s (M BDT)" % (
                r["stock_value_test"] / 1e6, (r["keep_all"] - r["oracle"]) / 1e6, (r["keep_all"] - r["rule_no_sale_3m"]) / 1e6,
                r["dynamic_saving_vs_keep"] / 1e6, [round(x / 1e6, 2) for x in r["dynamic_saving_ci"]],
                r["structural_saving_vs_keep"] / 1e6, [round(x / 1e6, 2) for x in r["structural_saving_ci"]],
                r["both_saving_vs_keep"] / 1e6, [round(x / 1e6, 2) for x in r["both_saving_ci"]],
                r["both_minus_dynamic"] / 1e6, [round(x / 1e6, 2) for x in r["both_minus_dynamic_ci"]]), flush=True)

json.dump(out, open(os.path.join(OUT, "decision_value.json"), "w"), indent=1)
print("saved")
