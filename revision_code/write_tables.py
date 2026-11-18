"""Write the LaTeX bodies of the controller, seed and sensitivity tables from outputs/sim3_summary.json
-> outputs/tab_controllers.tex, tab_seeds.tex, tab_sens.tex (pasted into responses/RES_*.tex by build)."""
import json, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
S = json.load(open(os.path.join(OUT, "sim3_summary.json")))
b = S["base"]


def n(x, d=1):
    s = f"{x:.{d}f}"
    return f"${s}$" if x < 0 else s


def ci(v):
    lo, hi = v["gain_ci"]; return f"[{n(lo)}, {n(hi)}]"


rows = [("B0 Firm's decisions", "B0 Firm's decisions", 0), ("B1 Rule-based ladder", "B1 Rule-based ladder", 0),
        ("B1h Rule-based ladder, firm's orders kept", "B1, firm's orders kept", 1),
        ("B2 Newsvendor", "B2 Newsvendor", 0), ("B2 Newsvendor, tuned on validation", "B2, tuned on validation", 1),
        ("B3 Newsvendor, risk-adjusted", "B3 Newsvendor, risk-adjusted", 0),
        ("B4 Periodic review (s,S)", "B4 Periodic review $(s,S)$", 0), ("B4 (s,S), tuned on validation", "B4, tuned on validation", 1),
        ("B3c Newsvendor + clearance ladder", "B3c Newsvendor + clearance ladder", 0),
        ("B4c (s,S) + clearance ladder", "B4c $(s,S)$ + clearance ladder", 0),
        ("B5 Multistage predict-then-optimize", "B5 Multistage planner", 0), ("B5, firm's orders kept", "B5, firm's orders kept", 1),
        ("B5, one month ahead", "B5, one month ahead", 1), ("B5, purchasing only", "B5, purchasing only", 1),
        ("B5, without structural covariates", "B5, without structural covariates", 1), None,
        ("QMIX", "QMIX (10 seeds)", 0), ("MAPPO", "MAPPO (10 seeds)", 0), ("MADDPG", "MADDPG (10 seeds)", 0),
        ("QMIX, two agents (automatic flag)", "QMIX, two agents (automatic flag)", 1),
        ("QMIX without structural covariates", "QMIX, without structural covariates", 1), None,
        ("Three-month foresight planner", "Three-month foresight planner (reference)", 0), ("Hindsight optimum", "Hindsight optimum (bound)", 0)]
L = []
for r in rows:
    if r is None: L.append("\\midrule"); continue
    k, lab, ind = r; v = b[k]; lab = ("\\quad " if ind else "") + lab
    if k.startswith("B0"):
        L.append(f"{lab} & {v['profit']:.1f} & -- & -- & -- & 0 & 0 \\\\"); continue
    liq = "--" if k.startswith("Hindsight") else f"{100 * v['liquidated']:.0f}"
    dis = "--" if k.startswith("Hindsight") else f"{100 * v['discount_share']:.0f}"
    L.append(f"{lab} & {v['profit']:.1f} & {n(v['gain'])} {ci(v)} & {100 * v['better']:.0f} & {100 * v['worse']:.0f} & {liq} & {dis} \\\\")
open(os.path.join(OUT, "tab_controllers.tex"), "w").write("\n".join(L) + "\n")

L = []
for k, lab in [("QMIX", "QMIX"), ("MAPPO", "MAPPO"), ("MADDPG", "MADDPG"), ("QMIX, two agents (automatic flag)", "QMIX, two agents"),
               ("QMIX without structural covariates", "QMIX, no structural covariates")]:
    v = b[k]; lo, hi = v["minus_B5_ci"]
    L.append(f"{lab} & {n(v['gain'])} & {v['gain_sd']:.1f} & {n(v['gain_min'])} to {n(v['gain_max'])} & {n(v['minus_B5'])} [{n(lo)}, {n(hi)}] \\\\")
open(os.path.join(OUT, "tab_seeds.tex"), "w").write("\n".join(L) + "\n")

SETS = ["base", "low lift", "high lift", "low recovery", "high recovery", "no decline", "capped terminal"]
L = []
for k, lab, withci in [("B1 Rule-based ladder", "B1 Rule-based ladder", False), ("B4c (s,S) + clearance ladder", "B4c $(s,S)$ + clearance", False),
                       ("B5 Multistage predict-then-optimize", "B5 Multistage planner", True), ("B5, stated parameters", "B5, base values", True),
                       ("QMIX", "QMIX", False), ("MAPPO", "MAPPO", False), ("MADDPG", "MADDPG", False), ("Hindsight optimum", "Hindsight optimum", True)]:
    vals = []
    for s in SETS:
        kk = "B5 Multistage predict-then-optimize" if (k == "B5, stated parameters" and s == "base") else k
        vals.append(S[s][kk])
    L.append(lab + " & " + " & ".join(n(v["gain"]) for v in vals) + " \\\\")
    if withci:
        L.append("\\quad 95\\% interval & " + " & ".join(ci(v) for v in vals) + " \\\\")
open(os.path.join(OUT, "tab_sens.tex"), "w").write("\n".join(L) + "\n")
print(open(os.path.join(OUT, "tab_controllers.tex")).read()); print(open(os.path.join(OUT, "tab_seeds.tex")).read()); print(open(os.path.join(OUT, "tab_sens.tex")).read())
