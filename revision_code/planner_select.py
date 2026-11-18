"""Choice of the planner's demand estimate on validation SKUs (base setting, risk estimates with
structural covariates): gain over the firm's decisions in million BDT -> outputs/planner_selection_oof.json"""
import json, os
from env2 import load_inputs
from sim2 import rollout, PARAMS
from controllers2 import b0, make_planner
D = load_inputs(); sk = D["skus"]; va = D["sets"]["val"]; PH = D["phat"]["both"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
tot = lambda pol: sum(rollout(sk[k], pol, PH[k], PARAMS)[0] for k in va)
base = tot(b0)
res = {d: (tot(make_planner(demand=d)) - base) / 1e6 for d in ["mean", "recent", "active"]}
json.dump(res, open(os.path.join(OUT, "planner_selection_oof.json"), "w"), indent=1); print(res)
