"""Exact attribution shares by covariate family for the risk classifier,
replicating the submitted SHAP step (SKU-level aggregation, SMOTE-Tomek on the
training SKUs, XGBoost on a 3,000-row subsample)."""
import json
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from common import *

CLF = ["StockVelocity_Scaled", "StockAccumRate_Scaled", "StockTrendSlope_Scaled", "CurrentStock_Scaled",
       "CurrentStkVal_Scaled", "OpeningStk_Scaled", "ClosingStkCM_Scaled", "MonthsStockNoSales_Scaled",
       "TotalSalesQty_Scaled", "TotalSalesVal_Scaled", "SalesQtyCM_Scaled", "SalesValCM_Scaled",
       "AvgMonthlySales_Scaled", "MaxMonthlySales_Scaled", "SalesCOV_Scaled", "MonthlySalesDensity_Scaled",
       "SalesDeclineSlope_Scaled", "ZeroSalesMonthsCount_Scaled", "MaxConsecZeroSalesMonths_Scaled",
       "TotalGRD_Scaled", "TotalGRDVAL_Scaled", "GRDCM_Scaled", "GRDVALCM_Scaled", "GRDRateCM_Scaled",
       "AddStockCM_Scaled", "DeductStockCM_Scaled", "CostPrice_Scaled", "RSP_Scaled", "GrossMarginPct_Scaled",
       "CapitalLockRatio_Scaled", "HoldingCostEstimate_Scaled", "CumSalesCoverageRatio_Scaled", "PCR_Scaled",
       "FreeToSalesRatio_Scaled", "ReturnRate_Scaled", "FreeQty_Scaled", "FreeVal_Scaled", "Month_sin",
       "Month_cos", "PGName_Enc", "MAGName_Enc", "ASGName_Enc", "AGName_Enc", "BrandDesc_Enc",
       "PriceFlag_Any_Scaled"]
STRUCT = ["PGName_Enc", "MAGName_Enc", "ASGName_Enc", "AGName_Enc", "BrandDesc_Enc"]
FIN = ["StockVelocity", "GrossMarginPct", "CapitalLockRatio", "HoldingCostEstimate", "PCR", "FreeToSales",
       "ReturnRate", "CostPrice", "RSP", "PriceFlag", "StockAccumRate", "GRDRate", "MonthlySalesDensity"]
CAL = ["Month_sin", "Month_cos"]
df = pd.read_csv(DATA).sort_values([SKU, MONTH])
tr, va, te = sku_split(df)
d = df[df[SKU].isin(tr)]
last = d.groupby(SKU).last().reset_index()
agg = d.groupby(SKU)[CLF].agg(["mean", "max", "std"]); agg.columns = ["_".join(c) for c in agg.columns]
m = last[[SKU, TARGET_NUM] + CLF].merge(agg.reset_index(), on=SKU)
names = CLF + list(agg.columns)
X = m[names].fillna(0).values.astype(np.float32); y = m[TARGET_NUM].values.astype(int)
Xs, ys = SMOTETomek(smote=SMOTE(k_neighbors=3, random_state=SEED), random_state=SEED).fit_resample(X, y)
rng = np.random.default_rng(SEED); idx = rng.choice(len(Xs), 3000, replace=False)
Xf, yf = Xs[idx], ys[idx]
cls, cnt = np.unique(yf, return_counts=True); w = {c: len(yf) / (3 * n) for c, n in zip(cls, cnt)}
clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7,
                        eval_metric="mlogloss", random_state=SEED, n_jobs=-1, objective="multi:softprob", num_class=3)
clf.fit(Xf, yf, sample_weight=np.array([w[v] for v in yf]))
C = clf.get_booster().predict(xgb.DMatrix(Xf[rng.choice(len(Xf), 1500, replace=False)]), pred_contribs=True)
a = np.abs(C[:, :, :-1]).mean(axis=(0, 1))  # exact TreeSHAP; mean |phi| over rows and classes


def fam(n):
    base = n.split("_Scaled")[0].split("_Enc")[0]
    if any(n.startswith(s) for s in STRUCT): return "structural"
    if any(n.startswith(c) for c in CAL): return "calendar"
    if any(base.startswith(f) for f in FIN): return "financial"
    return "demand_inventory"


fams = np.array([fam(n) for n in names])
share_all = {f: float(a[fams == f].sum() / a.sum()) for f in np.unique(fams)}
top = np.argsort(-a)[:22]
share_top = {f: float(a[top][fams[top] == f].sum() / a[top].sum()) for f in np.unique(fams)}
out = {"share_all_180": share_all, "share_top22": share_top,
       "top22": [(names[i], float(a[i]), fams[i]) for i in top]}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(OUT, "shap_share.json"), "w"), indent=1)
