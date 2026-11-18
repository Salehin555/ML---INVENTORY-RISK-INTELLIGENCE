"""Time-to-inactivity forecast (months since last sale at t+h) on the SKU-level
split used everywhere else, against two naive forecasts:
  persistence : MSLS_{t+h} = MSLS_t + h    (item stays inactive)
  calendar    : MSLS_{t+h} = month_{t+h}   (item never sells in the year)
Reported for all test SKUs and for test SKUs with at least one sale in the year."""
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from common import *

REG = ["CurrentStock_Scaled", "CurrentStkVal_Scaled", "OpeningStk_Scaled", "ClosingStkCM_Scaled",
       "AddStockCM_Scaled", "DeductStockCM_Scaled", "SalesQtyCM_Scaled", "SalesValCM_Scaled",
       "MonthlySalesDensity_Scaled", "GRDCM_Scaled", "GRDVALCM_Scaled", "GRDRateCM_Scaled",
       "CostPrice_Scaled", "RSP_Scaled", "GrossMarginPct_Scaled", "CapitalLockRatio_Scaled",
       "HoldingCostEstimate_Scaled", "PCR_Scaled", "FreeToSalesRatio_Scaled", "ReturnRate_Scaled",
       "StockVelocity_Scaled", "StockAccumRate_Scaled", "Month_sin", "Month_cos", "PGName_Enc",
       "MAGName_Enc", "ASGName_Enc", "AGName_Enc", "BrandDesc_Enc", "PriceFlag_Any_Scaled",
       "FreeQty_Scaled", "FreeVal_Scaled"]
df = pd.read_csv(DATA).sort_values([SKU, MONTH])
tr, va, te = sku_split(df)
active = set(df[df.SalesQtyCM > 0][SKU])

# correlation gate on training SKUs only (last-month snapshot)
last = df[df[SKU].isin(tr)].groupby(SKU).last()
corr = {c: abs(np.corrcoef(last[c], last.MonthsSinceLastSale)[0, 1]) for c in REG if last[c].std() > 0}
safe = [c for c in REG if corr.get(c, 0) <= 0.70]
print("gate: kept", len(safe), "max |r| = %.3f" % max(corr.values()))


def samples(skus, h):
    X, y, pers, cal, act = [], [], [], [], []
    for k, g in df[df[SKU].isin(skus)].groupby(SKU, sort=True):
        F = g[safe].values.astype(np.float32); m = g[MONTH].values; ms = g.MonthsSinceLastSale.values
        for t in range(len(g) - h):
            lag = F[t - 1] if t > 0 else np.zeros_like(F[t])
            X.append(np.concatenate([F[t], lag, [m[t] / 12]])); y.append(ms[t + h])
            pers.append(ms[t] + h); cal.append(m[t + h]); act.append(k in active)
    return [np.array(a) for a in (X, y, pers, cal, act)]


out = {"max_abs_corr": float(max(corr.values())), "n_safe": len(safe)}
for h in [3, 6, 11]:
    Xa, ya, _, _, _ = samples(tr, h)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1).fit(Xa, ya)
    X, y, pers, cal, act = samples(te, h); p = rf.predict(X)
    for sub, m in [("all", np.ones(len(y), bool)), ("sold_in_year", act)]:
        for name, pred in [("RandomForest", p), ("Persistence", pers), ("Calendar", cal)]:
            out[f"H{h}|{sub}|{name}"] = {
                "mae": float(mean_absolute_error(y[m], pred[m])),
                "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))),
                "r2": float(r2_score(y[m], pred[m])), "n": int(m.sum())}
            print(h, sub, name, {k: round(v, 3) for k, v in out[f"H{h}|{sub}|{name}"].items()})
json.dump(out, open(os.path.join(OUT, "forecast_check.json"), "w"), indent=1)
