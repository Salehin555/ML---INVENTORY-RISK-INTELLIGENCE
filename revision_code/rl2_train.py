"""Train one (algorithm, seed) configuration in the ledger simulator.
usage: python rl2_train.py ALGO SEED [EPISODES]"""
import sys, json, time, copy
from collections import deque
import torch.nn.functional as F
from torch.distributions import Categorical
from common import *

torch.set_num_threads(1)
ALGO, SEED_ = sys.argv[1], int(sys.argv[2])
N_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
PKEY = sys.argv[4] if len(sys.argv) > 4 else "both"
RWD = sys.argv[5] if len(sys.argv) > 5 else "per_sku"
AUTO = len(sys.argv) > 6 and sys.argv[6] == "auto"
TAG = (f"{ALGO}_s{SEED_}" if PKEY == "both" else f"{ALGO}-{PKEY}_s{SEED_}")
if RWD == "bdt": TAG = TAG.replace("_s", "-bdt_s")
if AUTO: TAG = TAG.replace("_s", "-auto_s")
MDIR = os.path.join(OUT, "models2"); os.makedirs(MDIR, exist_ok=True)
GAMMA = 0.99
from env2 import LedgerEnv, load_inputs
D = load_inputs()
set_seeds(SEED_)
env = LedgerEnv(D["sets"]["train"], D["skus"], D["phat"][PKEY], np.random.default_rng(SEED_), reward=RWD, autoflag=AUTO)
OBS = [SD, SD + MA, SD + MA]; ACT = [MA, PA, LA]


def oh(a, n=MA):
    v = np.zeros(n, np.float32); v[a] = 1; return v


def mlp(i, o, h=128, ln=False):
    L = [nn.Linear(i, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, h)] + \
        ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*L)


curve = []
t0 = time.time()

# ======================================================================= QMIX
if ALGO == "qmix":
    LR, BUF, BS, TGT, EMB = 5e-4, 80_000, 256, 200, 64

    class Mixer(nn.Module):
        def __init__(s):
            super().__init__()
            s.w1 = nn.Sequential(nn.Linear(SD, EMB), nn.ReLU(), nn.Linear(EMB, 3 * EMB))
            s.b1 = nn.Linear(SD, EMB)
            s.w2 = nn.Sequential(nn.Linear(SD, EMB), nn.ReLU(), nn.Linear(EMB, EMB))
            s.b2 = nn.Sequential(nn.Linear(SD, EMB), nn.ReLU(), nn.Linear(EMB, 1))

        def forward(s, q, st):
            B = q.size(0)
            h = F.elu(torch.bmm(q.unsqueeze(1), torch.abs(s.w1(st)).view(B, 3, EMB)) + s.b1(st).view(B, 1, EMB))
            return (torch.bmm(h, torch.abs(s.w2(st)).view(B, EMB, 1)) + s.b2(st).view(B, 1, 1)).view(B, 1)

    Q = [mlp(o, a) for o, a in zip(OBS, ACT)]; Qt = copy.deepcopy(Q)
    mix = Mixer(); mixt = copy.deepcopy(mix)
    params = [p for q in Q for p in q.parameters()] + list(mix.parameters())
    opt = torch.optim.Adam(params, lr=LR); buf = deque(maxlen=BUF)

    def greedy(s):
        with torch.no_grad():
            a0 = int(Q[0](torch.from_numpy(s[None].astype(np.float32))).argmax())
            o = np.concatenate([s, oh(a0)]).astype(np.float32)[None]
            return a0, int(Q[1](torch.from_numpy(o)).argmax()), int(Q[2](torch.from_numpy(o)).argmax())

    for ep in range(1, N_EP + 1):
        eps = max(0.05, 0.9995 ** ep)
        s = env.reset(); R = Rop = 0.0
        for _ in range(T):
            g = greedy(s)
            a0 = random.randrange(MA) if random.random() < eps else g[0]
            o12 = np.concatenate([s, oh(a0)]).astype(np.float32)
            if random.random() < eps: a1 = random.randrange(PA)
            else:
                with torch.no_grad(): a1 = int(Q[1](torch.from_numpy(o12[None])).argmax())
            if random.random() < eps: a2 = random.randrange(LA)
            else:
                with torch.no_grad(): a2 = int(Q[2](torch.from_numpy(o12[None])).argmax())
            ns, r, done, info = env.step(a0, a1, a2)
            R += r; Rop += info["operating"]
            buf.append((s.astype(np.float32), o12, (a0, a1, a2), r, ns.astype(np.float32), float(done)))
            s = ns
            if done: break
        curve.append((R, Rop))
        if len(buf) >= BS:
            b = random.sample(buf, BS)
            s0 = torch.tensor(np.array([x[0] for x in b])); o1 = torch.tensor(np.array([x[1] for x in b]))
            a = torch.tensor(np.array([x[2] for x in b])); r = torch.tensor([x[3] for x in b]).float().unsqueeze(1)
            s1 = torch.tensor(np.array([x[4] for x in b])); dn = torch.tensor([x[5] for x in b]).float().unsqueeze(1)
            qs = torch.cat([Q[0](s0).gather(1, a[:, :1]), Q[1](o1).gather(1, a[:, 1:2]), Q[2](o1).gather(1, a[:, 2:3])], 1)
            qtot = mix(qs, s0)
            with torch.no_grad():
                n0 = Q[0](s1).argmax(1, keepdim=True)
                no = torch.cat([s1, F.one_hot(n0.squeeze(1), MA).float()], 1)
                n1 = Q[1](no).argmax(1, keepdim=True); n2 = Q[2](no).argmax(1, keepdim=True)
                nq = torch.cat([Qt[0](s1).gather(1, n0), Qt[1](no).gather(1, n1), Qt[2](no).gather(1, n2)], 1)
                y = r + GAMMA * mixt(nq, s1) * (1 - dn)
            loss = F.mse_loss(qtot, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 10.0); opt.step()
        if ep % TGT == 0:
            for q, qt in zip(Q, Qt): qt.load_state_dict(q.state_dict())
            mixt.load_state_dict(mix.state_dict())
    torch.save({"Q": [q.state_dict() for q in Q]}, os.path.join(MDIR, TAG + ".pt"))

# ====================================================================== MAPPO
elif ALGO == "mappo":
    ROLL, EPOCHS, MB, CLIP, LAM, ENT, VC, GN = 48, 6, 64, 0.2, 0.95, 0.05, 0.5, 0.5
    actors = [mlp(o, a, ln=True) for o, a in zip(OBS, ACT)]
    critic = nn.Sequential(nn.Linear(sum(OBS), 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256),
                           nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    oa = [torch.optim.Adam(a.parameters(), lr=3e-4, eps=1e-5) for a in actors]
    oc = torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)
    B = {k: [] for k in ["o0", "o1", "g", "a", "lp", "r", "d", "v"]}
    ep = 0
    while ep < N_EP:
        s = env.reset(); R = Rop = 0.0
        for _ in range(T):
            with torch.no_grad():
                st = torch.from_numpy(s[None].astype(np.float32))
                d0 = Categorical(logits=actors[0](st)); a0 = d0.sample()
                o1 = torch.cat([st, F.one_hot(a0, MA).float()], 1)
                d1 = Categorical(logits=actors[1](o1)); a1 = d1.sample()
                d2 = Categorical(logits=actors[2](o1)); a2 = d2.sample()
                g = torch.cat([st, o1, o1], 1); v = critic(g).item()
            ns, r, done, info = env.step(int(a0), int(a1), int(a2))
            R += r; Rop += info["operating"]
            B["o0"].append(st[0]); B["o1"].append(o1[0]); B["g"].append(g[0])
            B["a"].append(torch.stack([a0[0], a1[0], a2[0]]))
            B["lp"].append(torch.stack([d0.log_prob(a0)[0], d1.log_prob(a1)[0], d2.log_prob(a2)[0]]))
            B["r"].append(r); B["d"].append(float(done)); B["v"].append(v)
            s = ns
            if done: break
        ep += 1; curve.append((R, Rop))
        if len(B["r"]) >= ROLL or ep == N_EP:
            n = len(B["r"]); adv = np.zeros(n, np.float32); gae = 0.0
            for t in reversed(range(n)):
                nv = 0.0 if t == n - 1 else B["v"][t + 1]
                delta = B["r"][t] + GAMMA * nv * (1 - B["d"][t]) - B["v"][t]
                gae = delta + GAMMA * LAM * (1 - B["d"][t]) * gae; adv[t] = gae
            ret = torch.tensor(adv + np.array(B["v"], np.float32)); adv = torch.tensor(adv)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            O0 = torch.stack(B["o0"]); O1 = torch.stack(B["o1"]); G = torch.stack(B["g"])
            A = torch.stack(B["a"]); LP = torch.stack(B["lp"])
            for _ in range(EPOCHS):
                idx = torch.randperm(n)
                for i in range(0, n, MB):
                    m = idx[i:i + MB]
                    cl = VC * F.mse_loss(critic(G[m]).squeeze(1), ret[m])
                    oc.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(critic.parameters(), GN); oc.step()
                    for k, (act, ob) in enumerate(zip(actors, [O0, O1, O1])):
                        dist = Categorical(logits=act(ob[m])); nlp = dist.log_prob(A[m, k])
                        ratio = torch.exp(nlp - LP[m, k])
                        lo = -torch.min(ratio * adv[m], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[m]).mean() - ENT * dist.entropy().mean()
                        oa[k].zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(act.parameters(), GN); oa[k].step()
            B = {k: [] for k in B}
    torch.save({"A": [a.state_dict() for a in actors]}, os.path.join(MDIR, TAG + ".pt"))

# ===================================================================== MADDPG
elif ALGO == "maddpg":
    BS, TAU, ENT0, ENT1, ENTD = 512, 0.005, 0.15, 0.01, 0.9993
    actors = [mlp(o, a, ln=True) for o, a in zip(OBS, ACT)]; at = copy.deepcopy(actors)
    TO, TA = sum(OBS), sum(ACT)
    crit = [nn.Sequential(nn.Linear(TO + TA, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, 1)) for _ in range(3)]
    ct = copy.deepcopy(crit)
    oa = [torch.optim.Adam(a.parameters(), lr=2e-4) for a in actors]
    oc = [torch.optim.Adam(c.parameters(), lr=5e-4) for c in crit]
    buf = deque(maxlen=200_000); ou = [np.zeros(a) for a in ACT]; ent = ENT0

    def act(k, o, explore):
        with torch.no_grad():
            p = F.softmax(actors[k](torch.from_numpy(o[None].astype(np.float32))), -1)[0].numpy()
        if explore:
            ou[k] = ou[k] + 0.15 * (0 - ou[k]) + 0.25 * np.random.standard_normal(len(ou[k])) * np.sqrt(0.1)
            p = np.exp(p + ou[k]); p /= p.sum()
        return int(np.argmax(p))

    for ep in range(1, N_EP + 1):
        ent = max(ENT1, ent * ENTD); s = env.reset(); R = Rop = 0.0; ou = [np.zeros(a) for a in ACT]
        for _ in range(T):
            a0 = act(0, s, True); o1 = np.concatenate([s, oh(a0)])
            a1 = act(1, o1, True); a2 = act(2, o1, True)
            ns, r, done, info = env.step(a0, a1, a2)
            R += r; Rop += info["operating"]
            no1 = np.concatenate([ns, oh(a0)])
            buf.append((np.concatenate([s, o1, o1]), np.concatenate([oh(a0), oh(a1, PA), oh(a2, LA)]), r,
                        np.concatenate([ns, no1, no1]), float(done)))
            s = ns
            if done: break
        curve.append((R, Rop))
        if len(buf) >= BS:
            b = random.sample(buf, BS)
            G = torch.tensor(np.array([x[0] for x in b]), dtype=torch.float32)
            Aj = torch.tensor(np.array([x[1] for x in b]), dtype=torch.float32)
            r = torch.tensor([x[2] for x in b]).float().unsqueeze(1)
            G1 = torch.tensor(np.array([x[3] for x in b]), dtype=torch.float32)
            dn = torch.tensor([x[4] for x in b]).float().unsqueeze(1)
            sl = [slice(0, SD), slice(SD, SD + SD + MA), slice(SD + SD + MA, TO)]
            with torch.no_grad():                    # next observations use the next monitoring action
                ns1 = G1[:, sl[0]]; p0 = F.softmax(at[0](ns1), -1)
                o1n = torch.cat([ns1, F.one_hot(p0.argmax(1), MA).float()], 1)
                G1 = torch.cat([ns1, o1n, o1n], 1)
                ta = torch.cat([p0, F.softmax(at[1](o1n), -1), F.softmax(at[2](o1n), -1)], 1)
            for k in range(3):
                with torch.no_grad():
                    y = r + GAMMA * ct[k](torch.cat([G1, ta], 1)) * (1 - dn)
                cl = F.mse_loss(crit[k](torch.cat([G, Aj], 1)), y)
                oc[k].zero_grad(); cl.backward(); oc[k].step()
                ca = torch.cat([F.softmax(actors[j](G[:, sl[j]]), -1) if j == k else
                                F.softmax(actors[j](G[:, sl[j]]), -1).detach() for j in range(3)], 1)
                pi = F.softmax(actors[k](G[:, sl[k]]), -1) + 1e-8
                al = -crit[k](torch.cat([G, ca], 1)).mean() + ent * (pi * torch.log(pi)).sum(-1).mean()
                oa[k].zero_grad(); al.backward(); nn.utils.clip_grad_norm_(actors[k].parameters(), 1.0); oa[k].step()
                for tp, p in zip(at[k].parameters(), actors[k].parameters()): tp.data.mul_(1 - TAU).add_(TAU * p.data)
                for tp, p in zip(ct[k].parameters(), crit[k].parameters()): tp.data.mul_(1 - TAU).add_(TAU * p.data)
    torch.save({"A": [a.state_dict() for a in actors]}, os.path.join(MDIR, TAG + ".pt"))

np.save(os.path.join(MDIR, TAG + "_curve.npy"), np.array(curve, np.float32))
print(TAG, "done in %.0fs" % (time.time() - t0), "last500 R=%.3f op=%.3f" % tuple(np.mean(curve[-500:], 0)))
