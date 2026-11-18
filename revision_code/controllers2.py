"""Non-learning controllers for the ledger simulator."""
import itertools
import numpy as np
from scipy.stats import norm
from sim2 import PARAMS, T0, T1, KAPPA, order_qty, REVIEW_COST

PL = np.array(list(itertools.product(range(4), range(4))))      # (purchase, clearance)


def past_stats(sku, t):
    past = sku.d[:t - 1]
    sold = np.where(past > 0)[0]
    msls = (t - 1 - (sold[-1] + 1)) if len(sold) else t - 1
    return past.mean(), past.std(), msls, len(sold) > 0


def nearest_order(sku, t, target):
    """purchasing action whose quantity is closest to target (stop, small, large)"""
    if target <= 0: return 0
    q = [order_qty(sku, t, P) for P in (0, 1, 2)]
    return int(np.argmin([abs(x - target) for x in q]))


def ladder(sku, t, q):
    mu, sd, msls, ever = past_stats(sku, t)
    if q > 0 and msls >= 6: return 3, 3          # flag high, liquidate
    if q > 0 and msls >= 3: return 2, 2          # flag medium, 20% discount
    return 0, 0


def b0(s, sku, t, q, ph, ex): return (0, 3, 0)                        # status quo


def b1(s, sku, t, q, ph, ex):                                          # rule-based ladder
    m, L = ladder(sku, t, q)
    return (m, 0 if L else 3, L)


def _nv(sku, t, q, cr):
    mu, sd, _, _ = past_stats(sku, t)
    if cr <= 0: return 0
    z = norm.ppf(min(max(cr, 1e-3), 1 - 1e-3))
    return nearest_order(sku, t, mu + z * sd - q)


def cr_b3(sku, ph):
    cu = sku.p - sku.c
    if cu <= 0: return 0.0
    co = ph * (1 - PARAMS["rho"]) * sku.c
    return cu / (cu + co) if cu + co > 0 else 1.0


def b2(s, sku, t, q, ph, ex): return (0, _nv(sku, t, q, 0.714), 0)                    # newsvendor
def b3(s, sku, t, q, ph, ex): return (0, _nv(sku, t, q, cr_b3(sku, ph)), 0)            # + forecast


def _ss(sku, t, q):
    mu, sd, _, _ = past_stats(sku, t)
    lo = mu + 1.645 * sd; hi = lo + mu
    if mu > 0 and q <= lo and hi - q > 0:
        return max(1, nearest_order(sku, t, hi - q))
    return 0


def b4(s, sku, t, q, ph, ex): return (0, _ss(sku, t, q), 0)                            # (s,S)


def b3c(s, sku, t, q, ph, ex):                                          # newsvendor + clearance ladder
    m, L = ladder(sku, t, q); return (m, _nv(sku, t, q, cr_b3(sku, ph)), L)


def b4c(s, sku, t, q, ph, ex):                                          # (s,S) + clearance ladder
    m, L = ladder(sku, t, q); return (m, _ss(sku, t, q), L)


def make_planner(H=3, oracle=False, allow_clear=True, prm=PARAMS, fixed_P=None, demand='mean'):
    """Multistage predict-then-optimize: forecast the next H months, enumerate all
    purchase/clearance sequences, apply the first decision, re-plan next month.
    Data-driven version: demand forecast (1 - p_hat) * mean past demand; stock left at
    the horizon valued at its expected value (1-p_hat) c + p_hat rho c e^{-beta T1}.
    Oracle version: recorded future demand and receipts, true end-of-year status."""
    def pol(s, sku, t, q, ph, ex):
        if ex: return (0, 0, 0)
        h = min(H, T1 - t + 1)
        acts = PL if allow_clear else PL[PL[:, 1] == 0]
        if fixed_P is not None: acts = acts[acts[:, 0] == fixed_P]
        idx = np.array(list(itertools.product(range(len(acts)), repeat=h)))
        mu = sku.d[:t - 1].mean(); gbar = sku.g[:t - 1].mean()
        if demand == 'recent': mu_f = sku.d[max(0, t - 4):t - 1].mean()
        elif demand == 'active':
            nz = np.where(sku.d[:t - 1] > 0)[0]
            mu_f = sku.d[nz[0]:t - 1].mean() if len(nz) else 0.0
        else: mu_f = mu
        N = len(idx); Q = np.full(N, float(q)); val = np.zeros(N); alive = np.ones(N, bool)
        for j in range(h):
            tt = t + j
            P, L = acts[idx[:, j], 0], acts[idx[:, j], 1]
            dem = sku.d[tt - 1] if oracle else (1 - ph) * mu_f
            hold = sku.g[tt - 1] if (oracle or j == 0) else gbar
            order = np.choose(P, [np.zeros(N), np.full(N, np.ceil(mu)), np.full(N, np.ceil(3 * mu)), np.full(N, hold)])
            adj = sku.adj[tt - 1] if (oracle or j == 0) else 0.0
            avail = np.maximum(Q + order + adj, 0.0)
            liq = (L == 3) & alive
            f = np.choose(np.minimum(L, 2), [np.ones(N), np.full(N, 0.9), np.full(N, 0.8)])
            lift = np.choose(np.minimum(L, 2), [np.ones(N), np.full(N, prm["lift10"]), np.full(N, prm["lift20"])])
            sales = np.minimum(avail, dem * lift)
            prof = np.where(liq, prm["rho"] * sku.c * np.exp(-prm["beta"] * (tt - 1)) * avail,
                            sales * sku.p * f - KAPPA / 12 * sku.c * (avail - sales)) - sku.c * order
            prof = prof - REVIEW_COST * (L > 0)
            val += np.where(alive, prof, 0.0)
            Q = np.where(liq | ~alive, 0.0, avail - sales); alive &= ~liq
        if oracle and t + h - 1 == T1:
            v_end = sku.c if sku.alive_end else prm["rho"] * sku.c * np.exp(-prm["beta"] * T1)
        else:
            v_end = (1 - ph) * sku.c + ph * prm["rho"] * sku.c * np.exp(-prm["beta"] * T1)
        val += Q * v_end
        P, L = acts[idx[np.argmax(val), 0]]
        return (3 if L == 3 else (2 if L in (1, 2) else 0), int(P), int(L))
    return pol


def make_nv(cr):
    """newsvendor with a given critical ratio (tuned on validation SKUs)"""
    return lambda s, sku, t, q, ph, ex: (0, _nv(sku, t, q, cr), 0)


def make_ss(z, k):
    """(s,S) with reorder point mu + z sd and order-up-to level s + k mu (tuned on validation SKUs)"""
    def pol(s, sku, t, q, ph, ex):
        mu, sd, _, _ = past_stats(sku, t)
        lo = mu + z * sd; hi = lo + k * mu
        if mu > 0 and q <= lo and hi - q > 0:
            return (0, max(1, nearest_order(sku, t, hi - q)), 0)
        return (0, 0, 0)
    return pol
