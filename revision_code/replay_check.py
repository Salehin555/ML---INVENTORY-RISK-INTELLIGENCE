"""Replay check: with the firm's own decisions (B0) the ledger simulator should reproduce the
recorded closing stock of November. Prints the share of test SKUs reproduced exactly."""
import json, os
import numpy as np, pandas as pd
from common import DATA
from env2 import load_inputs
from sim2 import step, T0, T1, PARAMS
D = load_inputs(); sk = D["skus"]; te = D["sets"]["test"]
cl = pd.read_csv(DATA, usecols=["ProductCode", "Month", "ClosingStkCM"]).pivot(index="ProductCode", columns="Month", values="ClosingStkCM")
ok = 0
for k in te:
    q, ex = sk[k].q4, False
    for t in range(T0, T1 + 1):
        _, q, ex = step(sk[k], t, q, (0, 3, 0), PARAMS, ex)
    ok += abs(q - cl.loc[k, T1]) < 1e-6
out = {"n": len(te), "exact": int(ok), "share": ok / len(te)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "replay_check.json"), "w"), indent=1); print(out)
