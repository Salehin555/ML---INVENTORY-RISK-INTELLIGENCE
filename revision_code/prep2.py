"""Inputs for the ledger simulator: SKU traces and month-by-month risk estimates.
p_hat[t] = estimated probability of no sale in months t..12, from an XGBoost model
trained on training and validation SKUs with information from months < t only
(two versions: dynamic covariates only, and dynamic plus structural)."""
import pickle
import xgboost as xgb
from common import *
from sim2 import SKU as SkuTrace, T0, T1

STRUCT = ["PGName_Enc", "MAGName_Enc", "ASGName_Enc", "AGName_Enc", "BrandDesc_Enc"]
DYN = ["SalesQtyCM", "SalesValCM", "GRDCM", "GRDVALCM", "AddStockCM", "DeductStockCM",
       "OpeningStk", "ClosingStkCM", "ClosingValCM", "CostPrice", "RSP", "GrossMarginPct", "PriceFlag_Any"]

df = pd.read_csv(DATA).sort_values([SKU, MONTH])
df = df[df.groupby(SKU)[MONTH].transform("count") == 12]          # full-year SKUs
tr, va, te = sku_split(pd.read_csv(DATA))
part = {**{k: "train" for k in tr}, **{k: "val" for k in va}, **{k: "test" for k in te}}
ids = np.array(sorted(df[SKU].unique()))
W = {c: df.pivot(index=SKU, columns=MONTH, values=c).loc[ids].values.astype(float) for c in DYN}
S = df.groupby(SKU)[STRUCT].first().loc[ids]


def feats(t):
    """features using months 1..t-1 plus opening stock of month t"""
    cols = {}
    for c in DYN:
        x = W[c][:, :t - 1]
        cols[c + "_last"] = x[:, -1]; cols[c + "_mean"] = x.mean(1); cols[c + "_max"] = x.max(1); cols[c + "_std"] = x.std(1)
    s = W["SalesQtyCM"][:, :t - 1]
    last_sale = np.array([np.where(r > 0)[0][-1] + 1 if (r > 0).any() else 0 for r in s])
    cols["msls"] = (t - 1) - last_sale; cols["zero_months"] = (s == 0).sum(1)
    cols["slope"] = np.polyfit(np.arange(t - 1), s.T, 1)[0] if t > 2 else np.zeros(len(s))
    cols["open_t"] = W["OpeningStk"][:, t - 1]
    return pd.DataFrame(cols, index=ids)


fit = np.array([part.get(k) in ("train", "val") for k in ids])
phat = {"dynamic": np.zeros((len(ids), T1 - T0 + 1)), "both": np.zeros((len(ids), T1 - T0 + 1))}
brier = {}; dead_share = {}
for t in range(T0, T1 + 1):
    F = feats(t)
    y = (W["SalesQtyCM"][:, t - 1:].sum(1) == 0).astype(int)
    for name, X in [("dynamic", F), ("both", F.join(S))]:
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=8, eval_metric="logloss")
        # test SKUs: model trained on all training and validation SKUs;
        # training and validation SKUs: out-of-fold predictions (5 folds by SKU)
        m.fit(X.values[fit], y[fit])
        pr = m.predict_proba(X.values)[:, 1]
        idx = np.where(fit)[0]; folds = np.random.default_rng(SEED).permutation(len(idx)) % 5
        for f in range(5):
            tr_i, ho_i = idx[folds != f], idx[folds == f]
            mf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, random_state=SEED, n_jobs=8, eval_metric="logloss")
            mf.fit(X.values[tr_i], y[tr_i]); pr[ho_i] = mf.predict_proba(X.values[ho_i])[:, 1]
        phat[name][:, t - T0] = pr
        if name == "both":
            te_m = np.array([part.get(k) == "test" for k in ids])
            brier[t] = (float(((pr[fit] - y[fit]) ** 2).mean()), float(((pr[te_m] - y[te_m]) ** 2).mean()))
    dead_share[t] = float(y.mean()); print("month", t, "dead share", y.mean().round(3), flush=True)

skus = {}
cost = df.groupby(SKU).CostPrice.median().loc[ids].values; rsp = df.groupby(SKU).RSP.median().loc[ids].values
for j, k in enumerate(ids):
    skus[k] = SkuTrace(W["SalesQtyCM"][j], W["GRDCM"][j], W["OpeningStk"][j], cost[j], rsp[j], W["AddStockCM"][j] - W["DeductStockCM"][j])
# decision-relevant SKUs: stock at month 4, or any sale or receipt in months 1-3 (known at month 4)
relevant = {k for j, k in enumerate(ids) if W["OpeningStk"][j, T0 - 1] > 0 or W["SalesQtyCM"][j, :T0 - 1].sum() > 0
            or W["GRDCM"][j, :T0 - 1].sum() > 0}
valid_price = {k for j, k in enumerate(ids) if cost[j] > 0 and rsp[j] > 0}
sets = {p: [k for k in ids if part.get(k) == p and k in relevant and k in valid_price] for p in ["train", "val", "test"]}
print({p: len(v) for p, v in sets.items()})
import json
json.dump({"brier_oof_vs_test": brier, "dead_share": dead_share}, open(os.path.join(OUT, "sim2_phat_check.json"), "w"), indent=1)
pickle.dump({"skus": skus, "phat": {n: dict(zip(ids, v)) for n, v in phat.items()}, "sets": sets},
            open(os.path.join(OUT, "sim2_inputs.pkl"), "wb"))
