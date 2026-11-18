"""Out-of-year test: train the risk classifier on 2024 training SKUs, score the
2024 test SKUs and the full 2025 snapshot against the firm's own 2025 labels.
Features are built identically in both years from the raw ERP columns
(tree models need no scaling); category codes use the 2024 mapping."""
import json
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from common import *

RAW = ["CostPrice", "RSP", "CurrentStock", "CurrentStkVal", "TotalGRD", "TotalSalesQty", "TotalSalesVal",
       "FreeQty", "SalesQtyCM", "SalesValCM", "GRDCM", "AddStockCM", "DeductStockCM", "OpeningStk", "ClosingStkCM"]
CAT = ["PGName", "MAGName", "ASGName", "AGName", "BrandDesc"]
LAB = {"Regular": 0, "Obsolete": 1, "EOL": 2}

d24 = pd.read_csv(DATA)
x = pd.ExcelFile(os.path.join(ROOT, "data 2025", "Product Cost_Status_Sales Value Database._Year 2025 (1).xlsx"))
mm = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
parts = []
for sh in x.sheet_names:
    p = pd.read_excel(x, sh); p[MONTH] = mm[sh[:3]]; parts.append(p)
d25 = pd.concat(parts).rename(columns={"ASGNAme": "ASGName"})
d25 = d25.drop_duplicates([SKU, MONTH])
for c in RAW: d25[c] = pd.to_numeric(d25[c], errors="coerce").fillna(0).clip(lower=0)
d25 = d25[d25.INVCat.isin(LAB)]
d25[TARGET_NUM] = d25.INVCat.map(LAB)
maps = {c: {v: i for i, v in enumerate(sorted(d24[c].astype(str).unique()))} for c in CAT}


def feats(d):
    d = d.sort_values([SKU, MONTH]); g = d.groupby(SKU)
    F = g[RAW].agg(["mean", "max", "std", "last"]); F.columns = ["_".join(c) for c in F.columns]
    F["zero_sale_months"] = g.SalesQtyCM.apply(lambda s: (s == 0).sum())
    F["n_months"] = g.size()
    last = g.last()
    for c in CAT: F[c] = last[c].astype(str).map(maps[c]).fillna(-1)
    return F.fillna(0), last[TARGET_NUM].astype(int)


tr, va, te = sku_split(d24)
F24, y24 = feats(d24); F25, y25 = feats(d25)
m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=SEED, n_jobs=8, eval_metric="mlogloss")
w = y24.loc[tr].map(len(tr) / (3 * y24.loc[tr].value_counts())).values
m.fit(F24.loc[tr], y24.loc[tr], sample_weight=w)


def score(F, y):
    P = m.predict_proba(F[F24.columns]); yp = P.argmax(1)
    return {"n": int(len(y)), "acc": accuracy_score(y, yp), "macro_f1": f1_score(y, yp, average="macro"),
            "f1_per_class": f1_score(y, yp, average=None).tolist(),
            "auc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "pred_share": np.bincount(yp, minlength=3).tolist(), "true_share": np.bincount(y, minlength=3).tolist()}


seen = F25.index.isin(d24[SKU].unique())
out = {"2024_test": score(F24.loc[te], y24.loc[te]), "2025_all": score(F25, y25),
       "2025_seen_in_2024": score(F25[seen], y25[seen]), "2025_new": score(F25[~seen], y25[~seen])}
# naive baselines: carry the 2024 label forward (seen SKUs); predict the 2024 majority class (new SKUs)
from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
yb = y24.loc[F25.index[seen]].values
out["naive_carry_forward_seen"] = {"acc": _acc(y25[seen], yb), "macro_f1": _f1(y25[seen], yb, average="macro")}
maj = int(y24.loc[tr].value_counts().idxmax())
out["naive_majority_new"] = {"acc": _acc(y25[~seen], np.full((~seen).sum(), maj)), "macro_f1": _f1(y25[~seen], np.full((~seen).sum(), maj), average="macro"), "class": maj}
mix = np.where(seen, y24.reindex(F25.index).fillna(maj).astype(int).values, maj)
out["naive_all"] = {"acc": _acc(y25, mix), "macro_f1": _f1(y25, mix, average="macro")}
# 2025 scores on the 2024 test SKUs only (no training SKU), and the stronger naive rule for new SKUs
te_set = set(te)
seen_te = F25.index.isin(list(te_set))
out["2025_on_2024_test_skus"] = score(F25[seen_te], y25[seen_te])
ybt = y24.loc[F25.index[seen_te]].values
out["naive_carry_forward_2024test"] = {"acc": _acc(y25[seen_te], ybt), "macro_f1": _f1(y25[seen_te], ybt, average="macro")}
out["naive_regular_new"] = {"acc": _acc(y25[~seen], np.zeros((~seen).sum(), int)), "macro_f1": _f1(y25[~seen], np.zeros((~seen).sum(), int), average="macro")}
mix2 = np.where(seen, y24.reindex(F25.index).fillna(0).astype(int).values, 0)
out["naive_all_regular_for_new"] = {"acc": _acc(y25, mix2), "macro_f1": _f1(y25, mix2, average="macro")}
# label stability for SKUs present in both years
both = sorted(set(F25.index) & set(F24.index))
out["label_same_both_years"] = float((y25.loc[both].values == y24.loc[both].values).mean())
out["n_both"] = len(both)
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(OUT, "out_of_year.json"), "w"), indent=1, default=float)
