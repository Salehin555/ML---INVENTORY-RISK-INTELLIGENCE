"""RQ1 figures: attribution (with like-for-like shares) and decision value by rule and horizon."""
import json, os
import numpy as np
from figstyle import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
fam_col = {"structural": BLUE, "financial": ORANGE, "demand_inventory": AQUA}
fam_lab = {"structural": "Structural", "financial": "Financial", "demand_inventory": "Sales and stock"}

NAMES = {"PGName": "Product group", "BrandDesc": "Brand", "ASGName": "Article sub-group", "AGName": "Article group",
         "MAGName": "Main article group", "SalesValCM": "Sales value", "GrossMarginPct": "Gross margin",
         "OpeningStk": "Opening stock", "MonthlySalesDensity": "Share of months with sales", "RSP": "Retail price",
         "TotalSalesVal": "Total sales value", "ClosingStkCM": "Closing stock", "StockVelocity": "Stock velocity",
         "CumSalesCoverageRatio": "Cumulative sales coverage", "CostPrice": "Unit cost", "PCR": "Price-to-cost ratio"}


def nice(n):
    agg = "mean" if n.endswith("_mean") else ("max" if n.endswith("_max") else "last")
    base = n.replace("_mean", "").replace("_max", "").replace("_Scaled", "").replace("_Enc", "")
    return f"{NAMES.get(base, base)} ({agg})"


S = json.load(open(os.path.join(OUT, "shap_share.json")))
R = json.load(open(os.path.join(OUT, "decision_value_robust.json")))
DV = json.load(open(os.path.join(OUT, "decision_value.json")))
fig = plt.figure(figsize=(6.0, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.55)
ax = fig.add_subplot(gs[0]); top = S["top22"][::-1]
ax.barh(range(len(top)), [t[1] for t in top], color=[fam_col[t[2]] for t in top], height=0.62)
ax.set_yticks(range(len(top))); ax.set_yticklabels([nice(t[0]) for t in top], fontsize=7)
ax.set_xlabel("Mean |SHAP| (classifier, all classes)"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=fam_col[f], label=fam_lab[f]) for f in fam_col], loc="lower right", fontsize=6.5)
panel(ax, "a", x=-0.05)
ax2 = fig.add_subplot(gs[1])
rows = [("Risk class, whole year\n(classifier, 180 columns)", S["share_all_180"]["structural"]),
        ("Risk class, month 6", DV["shap_structural_share_t0_6"]),
        ("No further sale, month 3", R["t0=3|horizon=2024"]["shap_structural_share_same_model"]),
        ("No further sale, month 6", R["t0=6|horizon=2024"]["shap_structural_share_same_model"]),
        ("No further sale, month 9", R["t0=9|horizon=2024"]["shap_structural_share_same_model"])][::-1]
y = np.arange(len(rows))
ax2.barh(y, [100 for _ in rows], color=GRID, height=0.55)
ax2.barh(y, [100 * r[1] for r in rows], color=BLUE, height=0.55)
for yi, r in zip(y, rows): ax2.text(100 * r[1] + 2, yi, f"{100 * r[1]:.0f}%", va="center", fontsize=7, color=INK)
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax2.set_xlim(0, 100)
ax2.set_xlabel("Structural share of attribution (%)"); ax2.grid(axis="y", visible=False)
panel(ax2, "b", x=-0.05)
save(fig, "fig_shap_families")

# decision value of structural covariates by rule and outcome horizon
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
specs = [("plug-in", BLUE, "o"), ("calibrated plug-in", ORANGE, "s"), ("tuned threshold", VIOLET, "^")]
for j, (hz, ttl) in enumerate([("2024", "Outcome measured to November 2024"), ("2024+2025", "Outcome extended with 2025 sales")]):
    a = axs[j]
    for i, (sp, c, mk) in enumerate(specs):
        xs = np.array([3, 6, 9]) + (i - 1) * 0.45
        v = [R[f"t0={t}|horizon={hz}"][f"{sp}|gain_struct_over_dyn"] / 1e6 for t in (3, 6, 9)]
        a.scatter(xs, v, color=c, marker=mk, s=26, zorder=3, label=sp.replace("plug-in", "plug-in rule").replace("calibrated plug-in rule", "calibrated rule"))
        if sp == "plug-in" and hz == "2024":
            ci = [DV[f"t0={t}|target=dead|rho=0.5|delta=0.08"]["both_minus_dynamic_ci"] for t in (3, 6, 9)]
            for x_, (lo, hi) in zip(xs, ci): a.plot([x_, x_], [lo / 1e6, hi / 1e6], color=c, lw=1.0)
    a.axhline(0, color=INK2, lw=0.7); a.set_xticks([3, 6, 9]); a.set_xticklabels(["Month 3", "Month 6", "Month 9"])
    a.set_title(ttl, fontsize=7.5, fontweight="normal", color=INK2)
    panel(a, "ab"[j])
axs[0].set_ylabel("Gain from adding structural\ncovariates (million BDT)")
lo = min(min(a.get_ylim()[0] for a in axs), -10); hi = max(a.get_ylim()[1] for a in axs)
for a in axs: a.set_ylim(lo - 0.5, hi + 0.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "fig_decision_value")
print("ok")
