"""Gym-style wrapper of the ledger simulator for the learning algorithms.
Reward = monthly profit / scale_i, scale_i = c * max(q4, 3 * mean demand in months 1-3, 1);
the end-of-horizon stock value is added to the last reward."""
import pickle, os
import numpy as np
from sim2 import step, terminal, features, T0, T1, PARAMS, min_flag

HERE = os.path.dirname(os.path.abspath(__file__))


def load_inputs():
    return pickle.load(open(os.path.join(HERE, "outputs", "sim2_inputs.pkl"), "rb"))


class LedgerEnv:
    def __init__(self, keys, skus, phat, rng, prm=PARAMS, reward='per_sku', autoflag=False):
        self.keys, self.skus, self.phat, self.rng, self.prm = keys, skus, phat, rng, prm
        self.reward = reward; self.autoflag = autoflag

    def reset(self, key=None):
        self.k = self.keys[self.rng.integers(len(self.keys))] if key is None else key
        self.sku = self.skus[self.k]; self.ph = self.phat[self.k]
        self.t, self.q, self.ex = T0, self.sku.q4, False
        self.scale = self.sku.c * max(self.sku.q4, 3 * self.sku.d[:T0 - 1].mean(), 1.0)
        return features(self.sku, self.t, self.q, self.ph[0], self.ex)

    def step(self, a_mon, a_pro, a_liq):
        if self.autoflag:              # flag set to the lowest level that allows the clearance action
            a_mon = min_flag(a_liq)
        r, self.q, self.ex = step(self.sku, self.t, self.q, (a_mon, a_pro, a_liq), self.prm, self.ex)
        done = self.t >= T1
        if done:
            r += terminal(self.sku, self.q, self.ex, self.prm)
        self.t += 1
        s = features(self.sku, min(self.t, T1), self.q, self.ph[min(self.t, T1) - T0], self.ex)
        if self.reward == 'bdt':      # profit in million BDT: the evaluation objective
            rr = float(np.clip(r / 1e6, -50, 50))
        else:                         # profit relative to the SKU's own scale
            rr = float(np.clip(r / self.scale, -5, 5))
        return s, rr, done, {"operating": rr, "profit": r}
