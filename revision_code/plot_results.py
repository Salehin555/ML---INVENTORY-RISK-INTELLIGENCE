"""Figures of Sections 5.3-5.5 (controllers, learning curves, sensitivity) from published results only:
outputs/sim3_summary.json and outputs/models2/*_curve.npy."""
import json, glob, os
import numpy as np, pandas as pd
from figstyle import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SETS = ["base", "low lift", "high lift", "low recovery", "high recovery", "no decline", "capped terminal"]
res = json.load(open(os.path.join(OUT, "sim3_summary.json"))); b = res["base"]

order = [("Hindsight optimum", "Hindsight optimum (bound)", GREEN), ("Three-month foresight planner", "Three-month foresight planner", GREEN),
         ("B1 Rule-based ladder", "B1 Rule-based ladder", NEUTRAL), ("B1h Rule-based ladder, firm's orders kept", "B1, firm's orders kept", NEUTRAL),
         ("B2 Newsvendor", "B2 Newsvendor", NEUTRAL), ("B3 Newsvendor, risk-adjusted", "B3 Newsvendor, risk-adjusted", NEUTRAL),
         ("B4 Periodic review (s,S)", "B4 Periodic review (s,S)", NEUTRAL), ("B3c Newsvendor + clearance ladder", "B3c Newsvendor + ladder", NEUTRAL),
         ("B4c (s,S) + clearance ladder", "B4c (s,S) + ladder", NEUTRAL), ("B5 Multistage predict-then-optimize", "B5 Multistage planner", ORANGE),
         ("B5, firm's orders kept", "B5, firm's orders kept", ORANGE), ("B5, one month ahead", "B5, one month ahead", ORANGE),
         ("QMIX", "QMIX", BLUE), ("MAPPO", "MAPPO", BLUE), ("MADDPG", "MADDPG", BLUE), ("QMIX, two agents (automatic flag)", "QMIX, two agents", BLUE)]
fig = plt.figure(figsize=(6.9, 4.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.08)
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax)
y = np.arange(len(order))[::-1]
for yi, (k, lab, c) in zip(y, order):
    v = b[k]; lo, hi = v["gain_ci"]
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="round")
    ax.scatter([v["gain"]], [yi], color=c, s=22, zorder=3, edgecolor="white", linewidth=0.6)
    if "seed_gains" in v: ax.scatter(v["seed_gains"], [yi] * len(v["seed_gains"]), s=5, color=c, alpha=0.45, zorder=2, linewidths=0)
    ax2.barh(yi, 100 * v["better"], color=BLUE, height=0.6); ax2.barh(yi, -100 * v["worse"], color=RED, height=0.6)
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([o[1] for o in order])
ax.set_xlabel("Gain over the firm's decisions (million BDT)"); ax.grid(axis="y", visible=False)
ax2.axvline(0, color=INK2, lw=0.8); ax2.set_xlabel("SKUs worse (left) / better (right), %"); ax2.grid(axis="y", visible=False)
plt.setp(ax2.get_yticklabels(), visible=False); ax2.tick_params(axis="y", length=0)
panel(ax, "a"); panel(ax2, "b", x=0.0)
save(fig, "fig_controllers")

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.1), sharey=True)
for j, (g, pre) in enumerate([("QMIX", "qmix_s"), ("MAPPO", "mappo_s"), ("MADDPG", "maddpg_s")]):
    C = np.array([pd.Series(np.load(f)[:, 0]).rolling(200, min_periods=50).mean().values for f in sorted(glob.glob(os.path.join(OUT, "models2", pre + "*_curve.npy")))])
    x = np.arange(C.shape[1]); q1, md, q3 = np.nanpercentile(C, [25, 50, 75], axis=0)
    axs[j].fill_between(x, q1, q3, color=BLUE, alpha=0.18, linewidth=0); axs[j].plot(x, md, color=BLUE, lw=1.2)
    axs[j].set_title(g, fontweight="normal", color=INK2); axs[j].set_xlabel("Training episode")
    panel(axs[j], "abc"[j])
axs[0].set_ylabel("Scaled profit per episode")
plt.tight_layout(); save(fig, "fig_learning_curves")

keys = [("B1 Rule-based ladder", "B1 Rule-based ladder"), ("B4c (s,S) + clearance ladder", "B4c (s,S) + ladder"),
        ("B5 Multistage predict-then-optimize", "B5 Multistage planner"), ("QMIX", "QMIX"), ("MAPPO", "MAPPO"), ("MADDPG", "MADDPG"),
        ("Hindsight optimum", "Hindsight optimum")]
labs = ["Base", "Low lift", "High lift", r"$\rho=0.3$", r"$\rho=0.7$", r"$\beta=0$", "Capped\nend value"]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
w = 0.11; xs = np.arange(len(SETS))
cols = [NEUTRAL, MAGENTA, ORANGE, BLUE, VIOLET, AQUA, YELLOW]
for i, ((k, lab), c) in enumerate(zip(keys, cols)):
    g = [res[s][k]["gain"] for s in SETS]; ci = np.array([res[s][k]["gain_ci"] for s in SETS])
    xx = xs + (i - 3) * w
    ax.vlines(xx, ci[:, 0], ci[:, 1], color=c, lw=1.0); ax.scatter(xx, g, color=c, s=14, zorder=3, label=lab, edgecolor="white", linewidth=0.4)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_ylabel("Gain over the firm (million BDT)")
ax.grid(axis="x", visible=False); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.30), fontsize=7.5, markerscale=1.6)
plt.tight_layout(); save(fig, "fig_sensitivity")
print("figures saved")
