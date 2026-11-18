"""Segmentation figures (appendix), regenerated from the data with the original settings."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from common import DATA, SKU, MONTH, TARGET_NUM, SEED, sku_split
from figstyle import *

COLS = ["StockVelocity", "ZeroSalesMonthsCount", "CapitalLockRatio", "HoldingCostEstimate", "GrossMarginPct", "PCR"]
df = pd.read_csv(DATA)
tr, va, te = sku_split(df)
td = df[df[SKU].isin(te)].sort_values(MONTH).groupby(SKU).last().reset_index()
X = td[COLS].fillna(0).values.astype(np.float32); y = td[TARGET_NUM].values.astype(int)
mn, mx = X.min(0), X.max(0); mx[mx == mn] = mn[mx == mn] + 1; Xn = (X - mn) / (mx - mn)
ks = range(2, 9); ine, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10); lb = km.fit_predict(Xn)
    ine.append(km.inertia_); sil.append(silhouette_score(Xn, lb))
km = KMeans(n_clusters=3, random_state=SEED, n_init=20); lab = km.fit_predict(Xn)
s3, d3 = silhouette_score(Xn, lab), davies_bouldin_score(Xn, lab)
order = np.argsort(km.cluster_centers_[:, 2]); tier = {order[0]: "Low", order[1]: "Medium", order[2]: "High"}
tiers = np.array([tier[l] for l in lab])
db = DBSCAN(eps=0.5, min_samples=5).fit_predict(Xn)
print("silhouette %.3f DB %.3f outliers %d" % (s3, d3, (db == -1).sum()))

# (a) inertia, (b) silhouette
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
ax[0].plot(list(ks), ine, "-o", color=BLUE, ms=4); ax[0].set_xlabel("Number of clusters $k$"); ax[0].set_ylabel("Within-cluster sum of squares")
ax[1].plot(list(ks), sil, "-o", color=BLUE, ms=4); ax[1].set_xlabel("Number of clusters $k$"); ax[1].set_ylabel("Silhouette score")
for a in ax: a.axvline(3, color=AQUA, lw=0.8, ls="--")
ax[1].annotate(f"$k=3$: {s3:.3f}", (3, s3), xytext=(3.4, 0.47), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.6))
panel(ax[0], "a"); panel(ax[1], "b"); plt.tight_layout(); save(fig, "fig_cluster_selection")

# composition of tiers by planners' class
names = {0: "Regular", 1: "Obsolete", 2: "EOL"}; cols = {"Regular": BLUE, "Obsolete": ORANGE, "EOL": AQUA}
ct = pd.crosstab(pd.Categorical(tiers, ["Low", "Medium", "High"]), pd.Series(y).map(names))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
for j, (data, ttl) in enumerate([(ct, "SKUs"), (ct.div(ct.sum(1), axis=0) * 100, "Share of tier (%)")]):
    left = np.zeros(len(data))
    for c in ["Regular", "Obsolete", "EOL"]:
        v = data[c].values if c in data else np.zeros(len(data))
        ax[j].barh(range(len(data)), v, left=left, color=cols[c], edgecolor="white", linewidth=1.0, height=0.6, label=c)
        left += v
    ax[j].set_yticks(range(len(data))); ax[j].set_yticklabels([f"{t} risk" for t in data.index]); ax[j].set_xlabel(ttl)
    ax[j].grid(axis="y", visible=False); ax[j].invert_yaxis()
ax[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22)); panel(ax[0], "a"); panel(ax[1], "b")
plt.tight_layout(); save(fig, "fig_cluster_composition")

# projection with DBSCAN outliers
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for t, c in [("High", RED), ("Low", BLUE), ("Medium", YELLOW)]:
    m = tiers == t; ax.scatter(Xn[m, 0], Xn[m, 2], s=8, color=c, alpha=0.75, linewidths=0, label=f"{t} risk")
m = db == -1; ax.scatter(Xn[m, 0], Xn[m, 2], s=40, marker="x", color=INK, linewidths=1.0, label="DBSCAN outlier")
ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlim(-2e-4, 1.3); ax.set_ylim(-2e-4, 1.3)
ax.set_xlabel("Stock velocity (scaled, symmetric log axis)"); ax.set_ylabel("Capital lock ratio (scaled, symmetric log axis)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6.5)
plt.tight_layout(); save(fig, "fig_cluster_projection")
