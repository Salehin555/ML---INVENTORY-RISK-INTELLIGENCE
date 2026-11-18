"""Evaluate all controllers in the ledger simulator (one setting per call).
usage: python evaluate3.py SETTING        (SETTING in SETTINGS below)
In every setting the planner and the hindsight bound use that setting's parameters;
'B5, stated parameters' keeps the base parameters; learned policies are those trained
in the base setting (not retrained)."""
import glob, json, sys, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from env2 import load_inputs
from sim2 import rollout, PARAMS, T0, T1, min_flag
from controllers2 import *
from hindsight import hindsight_value

torch.set_num_threads(1)
D = load_inputs(); sk = D["skus"]; te = D["sets"]["test"]; va = D["sets"]["val"]; PH = D["phat"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SETTINGS = {"base": dict(PARAMS),
            "low lift": dict(PARAMS, lift10=1.25, lift20=1.5),
            "high lift": dict(PARAMS, lift10=1.75, lift20=2.5),
            "low recovery": dict(PARAMS, rho=0.3),
            "high recovery": dict(PARAMS, rho=0.7),
            "no decline": dict(PARAMS, beta=0.0),
            "capped terminal": dict(PARAMS, cover=3)}


def mlp(i, o, h=128, ln=False):
    L = [nn.Linear(i, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, h)] + \
        ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*L)


def load_policy(path, auto=False):
    ck = torch.load(path); ln = "A" in ck; nets = []
    for sd, (o, a) in zip(ck["Q"] if "Q" in ck else ck["A"], [(10, 4), (14, 4), (14, 4)]):
        m = mlp(o, a, ln=ln); m.load_state_dict(sd); m.eval(); nets.append(m)

    def pol(s, sku, t, q, ph, ex):
        with torch.no_grad():
            st = torch.from_numpy(s[None])
            a0 = int(nets[0](st).argmax())
            o = torch.cat([st, F.one_hot(torch.tensor([a0]), 4).float()], 1)
            a1, a2 = int(nets[1](o).argmax()), int(nets[2](o).argmax())
            return (min_flag(a2) if auto else a0, a1, a2)
    return pol


def b1h(s, sku, t, q, ph, ex):
    m, L = ladder(sku, t, q); return (m, 3, L)


def guarded(prm, V, k):
    """Planner with a safeguard: items whose stock is worth more than V or that sold within
    the last k months are left to the firm's own decisions (routed to planner review)."""
    full = make_planner(demand="recent", prm=prm)
    def pol(s, sku, t, q, ph, ex):
        _, _, msls, _ = past_stats(sku, t)
        if sku.c * q > V or msls < k:
            return b0(s, sku, t, q, ph, ex)
        return full(s, sku, t, q, ph, ex)
    return pol


def total(pol, keys, key, prm):
    return sum(rollout(sk[k], pol, PH[key][k], prm)[0] for k in keys)


GRID_V = [np.inf, 5e6, 2e6, 1e6, 5e5, 2e5, 1e5]; GRID_K = [0, 1, 3, 6]
gpath = os.path.join(OUT, "guard_choice.json")
if not os.path.exists(gpath):                      # tuned once, on validation SKUs, base setting
    b0v = total(b0, va, "both", PARAMS); res = {}
    for V in GRID_V:
        for k in GRID_K:
            res[f"{V}|{k}"] = (total(guarded(PARAMS, V, k), va, "both", PARAMS) - b0v) / 1e6
    best = max(res, key=res.get)
    json.dump({"validation_gain": res, "best": best}, open(gpath, "w"), indent=1)
G = json.load(open(gpath)); Vb, kb = G["best"].split("|"); Vb, kb = float(Vb), int(kb)

cpath = os.path.join(OUT, "classical_choice.json")
if not os.path.exists(cpath):                      # classical parameters tuned once, on validation SKUs, base setting
    b0v = total(b0, va, "both", PARAMS)
    nv = {str(cr): (total(make_nv(cr), va, "both", PARAMS) - b0v) / 1e6 for cr in [0.3, 0.5, 0.6, 0.714, 0.8, 0.9, 0.95]}
    ss = {f"{z}|{k}": (total(make_ss(z, k), va, "both", PARAMS) - b0v) / 1e6
          for z in [0.0, 0.5, 1.0, 1.645, 2.33] for k in [0.5, 1, 2, 3]}
    json.dump({"newsvendor": nv, "ss": ss, "best_nv": max(nv, key=nv.get), "best_ss": max(ss, key=ss.get)}, open(cpath, "w"), indent=1)
CC = json.load(open(cpath)); CRb = float(CC["best_nv"]); Zb, Kb = map(float, CC["best_ss"].split("|"))
if sys.argv[1] == "tune": sys.exit()


def eff(A):
    """clearance actions that took effect: allowed by the flag and taken before exit"""
    L = A[:, 2].copy(); m = A[:, 0]
    L[((L == 1) | (L == 2)) & (m < 2)] = 0; L[(L == 3) & (m < 3)] = 0; L[A[:, 3] == 1] = 0
    return L


def rec(lab, k, tot, A):
    L = eff(A); alive = A[:, 3] == 0
    return (lab, k, tot, (L == 3).any(), ((L == 1) | (L == 2)).mean(), ((A[:, 1] == 0) & alive).sum() / max(alive.sum(), 1),
            (A[:, 2] == 3).any(), ((A[:, 2] == 1) | (A[:, 2] == 2)).mean())

name = sys.argv[1]; prm = SETTINGS[name]
ctrl = [("B0 Firm's decisions", b0, "both"), ("B1 Rule-based ladder", b1, "both"),
        ("B1h Rule-based ladder, firm's orders kept", b1h, "both"),
        ("B2 Newsvendor", b2, "both"), ("B3 Newsvendor, risk-adjusted", b3, "both"),
        ("B4 Periodic review (s,S)", b4, "both"),
        ("B3c Newsvendor + clearance ladder", b3c, "both"), ("B4c (s,S) + clearance ladder", b4c, "both"),
        ("B5 Multistage predict-then-optimize", make_planner(demand="recent", prm=prm), "both"),
        ("B5g Planner with safeguard", guarded(prm, Vb, kb), "both"),
        ("Three-month foresight planner", make_planner(oracle=True, prm=prm), "both")]
if name == "base":
    ctrl += [("B5, firm's orders kept", make_planner(demand="recent", fixed_P=3, prm=prm), "both"),
             ("B5, one month ahead", make_planner(H=1, demand="recent", prm=prm), "both"),
             ("B5, purchasing only", make_planner(allow_clear=False, demand="recent", prm=prm), "both"),
             ("B5, without structural covariates", make_planner(demand="recent", prm=prm), "dynamic"),
             ("B2 Newsvendor, tuned on validation", make_nv(CRb), "both"),
             ("B4 (s,S), tuned on validation", make_ss(Zb, Kb), "both")]
else:
    ctrl += [("B5, stated parameters", make_planner(demand="recent", prm=PARAMS), "both")]

recs, flags, logs = [], {}, {}
for lab, pol, key in ctrl:
    for k in te:
        tot, acts = rollout(sk[k], pol, PH[key][k], prm)
        A = np.array(acts); logs[(lab, k)] = A
        recs.append(rec(lab, k, tot, A))
    print(name, lab, "%.2f" % (sum(r[2] for r in recs if r[0] == lab) / 1e6), flush=True)
for k in te:
    recs.append(("Hindsight optimum", k, hindsight_value(sk[k], prm), False, 0.0, 0.0, False, 0.0))
print(name, "hindsight done", flush=True)
for p in sorted(glob.glob(os.path.join(OUT, "models2", "*.pt"))):
    tag = os.path.basename(p)[:-3]; key = "dynamic" if "-dynamic" in tag else "both"
    pol = load_policy(p, auto="-auto" in tag); cnt = np.zeros(4)
    for k in te:
        tot, acts = rollout(sk[k], pol, PH[key][k], prm)
        A = np.array(acts); logs[(tag, k)] = A
        recs.append(rec(tag, k, tot, A))
        act = A[A[:, 3] == 0]                      # months before exit
        cnt += np.bincount(act[:, 0].astype(int), minlength=4)
    flags[tag] = (cnt / cnt.sum()).tolist()
df = pd.DataFrame(recs, columns=["policy", "sku", "profit", "liquidated", "discount_share", "stop_share",
                                 "liquidation_chosen", "discount_chosen"])
if name == "base":
    import pickle; pickle.dump(logs, open(os.path.join(OUT, "sim3_actions_base.pkl"), "wb"))
df.to_csv(os.path.join(OUT, f"sim3_eval_{name.replace(' ', '_')}.csv"), index=False)
json.dump(flags, open(os.path.join(OUT, f"sim3_flags_{name.replace(' ', '_')}.json"), "w"), indent=1)
print(name, "saved", flush=True)
