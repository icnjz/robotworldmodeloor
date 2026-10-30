"""Analysis of the restricted-vs-broad pair
Windows: 64 central_val + 98 out-of-range high_dev windows, 2 draws each, predictions on the 120 grid as pr[1:, ::2, ::2].

Per high window:
 excess log( late M-MSE of R / late M-MSE of B ), draws averaged (late = frames 25-36); also on frames after the first exceedance
 action dist amp_ratio = max_{t in window} ||a_t|| / core bound; excess_mass = sum_t max(0, ||a_t|| - bound) / bound
 state dist mean over true late frames of the nearest-neighbour RMS distance (40 x 40 RGB, area-downsampled) to a library of core_train frames
 (every 2nd frame of every core_train episode), divided by the median of the same quantity over central windows
 support pull share of late frames where the prediction is closer to some core-library frame than to its own truth, for R and for B

Correlations: Spearman of excess with action and state distance, and partial Spearman (rank residuals) of each controlling for the other; bootstrap CI.

Seed decomposition: late bias and seed variance for R and B on central and high windows.


Usage: SPLIT=... ROBOT_DATA=... OUT=<dir with clean_eval_R.npz, clean_eval_B.npz> python pair_analysis.py -> <OUT>/pair_analysis.{json,txt}"""
import os, json, numpy as np, cv2, av, torch
OUT = os.environ["OUT"]; S = json.load(open(os.environ["SPLIT"])); DATA = os.environ["ROBOT_DATA"]; G = S["groups"]; bound = float(S["core_amplitude_bound"])
metas = torch.load(os.path.join(DATA, "metadata.pt"), weights_only=False); torch.set_num_threads(int(os.environ.get("THREADS", 16)))
ZR, ZB = np.load(os.path.join(OUT, "clean_eval_R.npz")), np.load(os.path.join(OUT, "clean_eval_B.npz"))
for k in ("win_group", "win_e", "win_s0"): assert np.array_equal(ZR[k], ZB[k]), k
grp, E, S0 = ZR["win_group"], ZR["win_e"], ZR["win_s0"]; n = len(E); L = 37
assert not (set(E.tolist()) & (set(G["high_sealed_v2"]) | set(G["sealed_v1_excluded"]) | set(G["core_train"]) | set(G["broad_support_train"])))
first_ex = {(w["e"], w["s0"]): w["first_exceed"] for w in S["window_exposure"]["windows"]}


def video(ti, s0=0, n_=None, size=240):
    out = []
    with av.open(os.path.join(DATA, os.path.basename(metas[ti]["video_path"]))) as c:
        for st in c.streams.video: st.thread_count = 1
        for f in c.decode(video=0):
            a = f.to_ndarray(format="rgb24")
            if a.shape[:2] != (size, size): a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
            out.append(a)
            if n_ is not None and len(out) >= s0 + n_: break
    v = np.stack(out)[s0:]
    if n_ is not None:
        v = v[:n_]
        if len(v) < n_: v = np.concatenate([v, np.repeat(v[-1:], n_ - len(v), 0)])
    return v.astype(np.float32) / 255


d40 = lambda X: np.stack([cv2.resize(x, (40, 40), interpolation=cv2.INTER_AREA) for x in X]).reshape(len(X), -1)
LIBN = int(os.environ.get("LIBN", 0)); CORE = G["core_train"][:LIBN] if LIBN else G["core_train"]   # LIBN only for the code test
lib = np.concatenate([d40(video(int(e))[::2]) for e in CORE]).astype(np.float32); LIB = torch.from_numpy(lib); L2 = (LIB ** 2).sum(1)
print("core library frames", lib.shape, flush=True)


def nn_dist(X):   # X (m, 4800) float -> RMS distance to the nearest library frame
    Xt = torch.from_numpy(X.astype(np.float32)); d2 = (Xt ** 2).sum(1, keepdim=True) + L2[None] - 2 * Xt @ LIB.T
    return np.sqrt(np.maximum(d2.min(1).values.numpy(), 0) / X.shape[1])


per = {k: np.zeros(n) for k in ("eR", "eB", "pR", "pB", "amp", "mass", "sd", "pullR", "pullB", "VR", "BR", "VB", "BB")}
for i in range(n):
    e, s0 = int(E[i]), int(S0[i]); A = np.linalg.norm(np.asarray(metas[e]["actions"], np.float64)[s0:s0 + L], axis=1)
    per["amp"][i] = A.max() / bound; per["mass"][i] = np.maximum(0, A - bound).sum() / bound
    gt = video(e, s0, L)[:, ::2, ::2]; y = gt[1:]; W = 0.01 + np.abs(gt - gt[:1]).max(-1).max(0); norm = W.sum() * 3
    wn = lambda X, lo, hi: float((W[None, ..., None] * X[lo:hi] ** 2).sum() / (norm * (hi - lo)))
    fe = max(first_ex.get((e, s0), 0), 1) if grp[i] == "high" else 1
    for tag, Z in (("R", ZR), ("B", ZB)):
        Y1, Y2 = Z["D1"][i].astype(np.float32) / 255, Z["D2"][i].astype(np.float32) / 255
        per["e" + tag][i] = (wn(Y1 - y, 24, 36) + wn(Y2 - y, 24, 36)) / 2; per["p" + tag][i] = (wn(Y1 - y, fe - 1, 36) + wn(Y2 - y, fe - 1, 36)) / 2
        V = wn(Y1 - Y2, 24, 36) / 2; per["V" + tag][i] = V; per["B" + tag][i] = wn((Y1 + Y2) / 2 - y, 24, 36) - V / 2
        dl = nn_dist(d40(Y1[24:36])); dt = np.sqrt(((d40(Y1[24:36]) - d40(y[24:36])) ** 2).mean(1)); per["pull" + tag][i] = float((dl < dt).mean())
    per["sd"][i] = float(nn_dist(d40(y[24:36])).mean())
    if i % 20 == 0: print("window", i + 1, n, flush=True)

hi_, ce = grp == "high", grp == "central"; per["sd"] = per["sd"] / np.median(per["sd"][ce])
rk = lambda x: np.argsort(np.argsort(x)).astype(float)
def spear(x, y): return float(np.corrcoef(rk(x), rk(y))[0, 1])
def partial(x, y, z):   # Spearman of x and y controlling for z (rank residuals)
    rx, ry, rz = rk(x), rk(y), rk(z); bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return float(np.corrcoef(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))[0, 1])
ix = np.random.default_rng(0).integers(0, hi_.sum(), (2000, hi_.sum())); H = {k: v[hi_] for k, v in per.items()}
ex = np.log(H["eR"] / H["eB"]); exp_ = np.log(H["pR"] / H["pB"]); out = {"n_high": int(hi_.sum()), "n_central": int(ce.sum())}; lines = []
def stat(name, f):
    v = f(np.arange(len(ex))); b = [f(J) for J in ix]; out[name] = {"value": float(v), "ci": np.percentile(b, [2.5, 97.5]).tolist()}
    lines.append(f"{name}: {v:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
stat("median_log_excess_late_high", lambda J: np.median(ex[J])); stat("median_log_excess_post_exceed_high", lambda J: np.median(exp_[J]))
stat("spearman_excess_vs_action_amp", lambda J: spear(ex[J], H["amp"][J])); stat("spearman_excess_vs_state_dist", lambda J: spear(ex[J], H["sd"][J]))
stat("partial_excess_action_given_state", lambda J: partial(ex[J], H["amp"][J], H["sd"][J])); stat("partial_excess_state_given_action", lambda J: partial(ex[J], H["sd"][J], H["amp"][J]))
stat("support_pull_R_minus_B_high", lambda J: H["pullR"][J].mean() - H["pullB"][J].mean())
for tag in ("R", "B"):
    for g, m in (("central", ce), ("high", hi_)):
        Vs, Bs = per["V" + tag][m].sum(), per["B" + tag][m].sum(); out[f"seed_{tag}_{g}"] = {"bias": float(Bs / m.sum()), "var": float(Vs / m.sum()), "var_share": float(Vs / (Vs + Bs))}
        lines.append(f"{tag} {g}: late bias {1e3 * Bs / m.sum():.3f} seed variance {1e3 * Vs / m.sum():.3f} (x1e-3), variance share {100 * Vs / (Vs + Bs):.1f}%")
lines.append(f"state distance of high windows (median, central = 1): {np.median(H['sd']):.2f}; action amp ratio median {np.median(H['amp']):.2f}; support pull R {H['pullR'].mean():.2f} B {H['pullB'].mean():.2f}")
np.savez(os.path.join(OUT, "pair_analysis_perwindow.npz"), **per, grp=grp, E=E, S0=S0)
json.dump(out, open(os.path.join(OUT, "pair_analysis.json"), "w"), indent=1); open(os.path.join(OUT, "pair_analysis.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
