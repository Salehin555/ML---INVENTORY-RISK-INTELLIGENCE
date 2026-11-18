"""Shared definitions for the revision experiments.

Faithful to the submitted pipeline (state variables, action effects, reward
weights, noise, blending), with three corrections:
  1. The surrogate transition model takes the state only.  Action effects are
     added afterwards by the stated deltas (the submitted model received an
     action encoding at rollout that it never saw during training).
  2. In QMIX the Procurement and Liquidation agents observe the Monitor agent's
     action, as in the other two algorithms and as described in the paper.
  3. Every controller is evaluated in the same simulator, on the same held-out
     SKUs and the same random draws (common random numbers).
"""
import os, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data 2024", "master_cleaned.csv")
OUT = os.path.join(ROOT, "revision_code", "outputs")
os.makedirs(OUT, exist_ok=True)

SEED = 42
SKU, MONTH, TARGET_NUM = "ProductCode", "Month", "INVCat_Num"

STATE_COLS = ["StockVelocity", "CapitalLockRatio", "HoldingCostEstimate",
              "GrossMarginPct", "SalesValCM", "CurrentStock", "PCR",
              "ZeroSalesMonthsCount", "GRDRateCM", "MonthlySalesDensity"]
SD = 10
MA = PA = LA = 4
T = 12
GAMMA = 0.95
NOISE = 0.05

# reward weights of the submitted model
RW_HCE, RW_CAP, RW_VEL = 8.0, 3.0, 2.0
RW_COORD, RW_IDLE, RW_OVERLIQ, RW_DIV, RW_TERM = 0.8, 0.20, 0.30, 0.25, 3.0
REPEAT_PEN = 0.12

MON_NAMES = ["No-Action", "Flag-Low", "Flag-Med", "Flag-High"]
PRO_NAMES = ["Stop", "Reorder-Small", "Reorder-Large", "Hold"]
LIQ_NAMES = ["Hold", "Disc-10%", "Disc-20%", "Liquidate"]


def set_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def sku_split(df):
    """Identical to the submitted pipeline: stratified 70/15/15 at SKU level."""
    from sklearn.model_selection import train_test_split
    sl = df.groupby(SKU)[TARGET_NUM].first().reset_index()
    sk, lb = sl[SKU].values, sl[TARGET_NUM].values
    tv, te, _, _ = train_test_split(sk, lb, test_size=0.15, stratify=lb,
                                    random_state=SEED)
    ltv = sl.set_index(SKU).loc[tv, TARGET_NUM].values
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=ltv,
                              random_state=SEED)
    return tr, va, te


class WorldModel(nn.Module):
    """State-only transition model: s_{t+1} = s_t + g(s_t) (autonomous dynamics)."""
    def __init__(self, sd=SD, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, sd))

    def forward(self, s):
        # residual form: predict the monthly change, so persistence is the default
        return s + self.net(s)


# ---------------------------------------------------------------- actions
def apply_action_effects(s, a_mon, a_pro, a_liq):
    """Stated action-effect deltas (unchanged from the submitted model)."""
    s = s.copy()
    if a_pro == 0:   s[1] = max(0, s[1]-.05); s[2] = max(0, s[2]-.04); s[5] = max(0, s[5]-.03)
    elif a_pro == 1: s[1] = min(1, s[1]+.02); s[2] = min(1, s[2]+.01); s[5] = min(1, s[5]+.04)
    elif a_pro == 2: s[1] = min(1, s[1]+.07); s[2] = min(1, s[2]+.05); s[5] = min(1, s[5]+.10)
    if a_liq == 1:   s[0] = min(1, s[0]+.07); s[2] = max(0, s[2]-.04); s[3] = max(0, s[3]-.02); s[5] = max(0, s[5]-.05)
    elif a_liq == 2: s[0] = min(1, s[0]+.14); s[2] = max(0, s[2]-.09); s[3] = max(0, s[3]-.05); s[5] = max(0, s[5]-.10); s[1] = max(0, s[1]-.04); s[4] = min(1, s[4]+.06)
    elif a_liq == 3: s[0] = min(1, s[0]+.22); s[2] = max(0, s[2]-.16); s[3] = max(0, s[3]-.10); s[5] = max(0, s[5]-.18); s[1] = max(0, s[1]-.09); s[4] = min(1, s[4]+.12)
    return np.clip(s, 0, 1)


def apply_action_effects_batch(S, a_pro, a_liq):
    """Vectorised version for planning. S: (n,10); a_pro, a_liq: (n,)"""
    S = S.copy()
    def add(mask, j, d):
        S[mask, j] = np.clip(S[mask, j] + d, 0, 1)
    m = a_pro == 0; add(m, 1, -.05); add(m, 2, -.04); add(m, 5, -.03)
    m = a_pro == 1; add(m, 1, .02); add(m, 2, .01); add(m, 5, .04)
    m = a_pro == 2; add(m, 1, .07); add(m, 2, .05); add(m, 5, .10)
    m = a_liq == 1; add(m, 0, .07); add(m, 2, -.04); add(m, 3, -.02); add(m, 5, -.05)
    m = a_liq == 2; add(m, 0, .14); add(m, 2, -.09); add(m, 3, -.05); add(m, 5, -.10); add(m, 1, -.04); add(m, 4, .06)
    m = a_liq == 3; add(m, 0, .22); add(m, 2, -.16); add(m, 3, -.10); add(m, 5, -.18); add(m, 1, -.09); add(m, 4, .12)
    return S


# ---------------------------------------------------------------- rewards
def operating_terms(s_old, s_new, a_pro, a_liq, done):
    """Cost-based part of the reward: holding cost, capital lock, sell-through,
    penalties for marking down healthy stock, over-liquidation and ending the
    horizon in a stranded state.  Contains no behaviour-shaping terms."""
    r = RW_HCE * (s_old[2] - s_new[2])
    r -= RW_CAP * max(0.0, s_new[1] - s_old[1])
    r += RW_VEL * max(0.0, s_new[0] - s_old[0])
    c = 0.0
    if a_liq >= 2 and s_old[0] > 0.60: c -= 0.35
    if a_liq >= 2 and s_old[7] < 0.20: c -= 0.30
    if a_liq == 3 and s_old[2] < 0.12: c -= 0.25
    if a_liq == 3 and s_old[0] > 0.50 and s_old[7] < 0.25: c -= 0.45
    if a_pro == 2 and s_old[5] < 0.05: c -= 0.20
    if a_liq == 3 and s_new[3] < 0.12: c -= RW_OVERLIQ
    if done and s_new[0] < 0.15 and s_new[2] > 0.35: c -= RW_TERM
    return r + c


def operating_terms_batch(S0, S1, a_pro, a_liq, done):
    r = RW_HCE * (S0[:, 2] - S1[:, 2])
    r -= RW_CAP * np.maximum(0, S1[:, 1] - S0[:, 1])
    r += RW_VEL * np.maximum(0, S1[:, 0] - S0[:, 0])
    r -= 0.35 * ((a_liq >= 2) & (S0[:, 0] > 0.60))
    r -= 0.30 * ((a_liq >= 2) & (S0[:, 7] < 0.20))
    r -= 0.25 * ((a_liq == 3) & (S0[:, 2] < 0.12))
    r -= 0.45 * ((a_liq == 3) & (S0[:, 0] > 0.50) & (S0[:, 7] < 0.25))
    r -= 0.20 * ((a_pro == 2) & (S0[:, 5] < 0.05))
    r -= RW_OVERLIQ * ((a_liq == 3) & (S1[:, 3] < 0.12))
    if done:
        r -= RW_TERM * ((S1[:, 0] < 0.15) & (S1[:, 2] > 0.35))
    return r


def shaping_terms(s_old, a_mon, a_pro, a_liq, action_counts):
    """Behaviour-shaping terms of the submitted training reward."""
    r = 0.0
    if a_mon == 0 and a_pro == 3 and a_liq == 0:
        r -= RW_IDLE
    for a_used, counts in zip([a_mon, a_pro, a_liq], action_counts):
        tot = sum(counts.values()) + 1e-6
        if max(counts.values(), default=0) / tot > 0.80:
            dom = max(counts, key=counts.get, default=a_used)
            if a_used != dom:
                r += RW_DIV
    if a_mon == 3 and a_pro == 0 and a_liq >= 1 and s_old[7] > 0.50 and s_old[2] > 0.25:
        r += RW_COORD
    return r


# ---------------------------------------------------------------- simulator
class InventorySim:
    """Surrogate simulator.  reward_mode: 'shaped' (submitted training reward)
    or 'operating' (cost terms only)."""
    def __init__(self, wm, seqs, reward_mode="shaped", rng=None):
        self.wm = wm; self.seqs = seqs; self.mode = reward_mode
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self, idx=None):
        self.k = self.rng.integers(len(self.seqs)) if idx is None else idx
        self.seq = self.seqs[self.k]; self.t = 0
        self.state = self.seq[0].copy()
        self.counts = [{} for _ in range(3)]; self.hist = [[] for _ in range(3)]
        return self.state.copy()

    def _wm(self, s):
        with torch.no_grad():
            return self.wm(torch.from_numpy(s[None].astype(np.float32))).numpy()[0]

    def step(self, a_mon, a_pro, a_liq):
        for i, a in enumerate([a_mon, a_pro, a_liq]):
            self.counts[i][a] = self.counts[i].get(a, 0) + 1
            self.hist[i].append(a)
        s1 = self._wm(self.state)
        s1 = apply_action_effects(s1, a_mon, a_pro, a_liq)
        s1 = s1 + self.rng.normal(0, NOISE, size=SD)
        s1[4] = np.clip(s1[4] * self.rng.uniform(0.8, 1.2), 0, 1)
        if self.rng.random() < 0.05:
            s1[5] = max(0.0, s1[5] - self.rng.uniform(0.02, 0.08))
        s1 = np.clip(s1, 0, 1)
        self.t += 1; done = self.t >= T
        op = operating_terms(self.state, s1, a_pro, a_liq, done)
        sh = shaping_terms(self.state, a_mon, a_pro, a_liq, self.counts)
        rep = sum(-REPEAT_PEN for h in self.hist if len(h) >= 3 and h[-1] == h[-2] == h[-3])
        shaped = float(np.clip(float(np.clip(op + sh, -2, 2)) + rep, -3, 3))
        info = {"operating": op, "shaped": shaped, "s_raw": s1}
        self.state = s1
        if self.t < len(self.seq):
            b = max(0.0, 0.3 - self.t * 0.03)
            self.state = b * self.seq[self.t] + (1 - b) * s1
        r = shaped if self.mode == "shaped" else float(np.clip(op, -3, 3))
        return self.state.copy(), r, done, info


def load_prep():
    d = np.load(os.path.join(OUT, "prep.npz"), allow_pickle=True)
    wm = WorldModel(); wm.load_state_dict(torch.load(os.path.join(OUT, "world_model.pt")))
    wm.eval()
    return d, wm
