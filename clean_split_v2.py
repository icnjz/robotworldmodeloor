"""Episode split by command amplitude (v2, used for all results). CPU, actions and first frames only; no model errors.

scores command
AMPLITUDE: s_e = max_t ||a_{e,t}||_2 in the commands' physical units (a command with norm above every core command lies outside the convex hull
of the core commands). Membership rules (whole episodes; the v1 held-out episodes stay held out and are excluded from everything):
 core s_e < p80 of the remaining episodes -> restricted training, minus a seeded 10% central_val
 buffer p80 <= s_e < p90 -> never in the restricted model
 high s_e >= p90 -> seeded 50 / 25 / 25: high_dev (evaluation), high_sealed_v2 (held out, never read), high_broad (broad-support training only)
 broad-support training = (core_train minus as many random core episodes as there are buffer + high_broad episodes) + buffer + high_broad,
 i.e. the same episode count as the restricted model; both exclude central_val, high_dev and all held-out episodes.

Window-level exposure audit for the evaluation windows (official window length 37; starts on a fixed grid every 9 frames): for every high_dev
window, whether a command above the core amplitude boundary (max core-train norm) is supplied in the window, the first such frame, the count,
and whether the conditioning frame precedes it (the out-of-range evaluation windows start before the first exceedance).

Configuration audit (initial frames, observables only): robot matte area and centroid of frame 0 vs bg_clean, core vs high (quantile comparison).

Usage: python clean_split_v2.py -> clean_split_robot_arm_v2.json"""
import os, json, hashlib, numpy as np, torch, av, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("ROBOT_DATA", os.path.join(HERE, "acwm", "data", "kinematics", "robot_arm_64", "ind_train"))
V1 = os.path.join(HERE, "clean_split_robot_arm.json"); v1 = json.load(open(V1)); v1_hash = hashlib.md5(open(V1, "rb").read()).hexdigest()
metas = torch.load(os.path.join(DATA, "metadata.pt"), weights_only=False); A = [np.asarray(m["actions"], np.float64) for m in metas]; nE = len(A)
sealed_v1 = set(v1["groups"]["high_sealed"]); elig = np.array([i for i in range(nE) if i not in sealed_v1])
s = np.array([np.linalg.norm(a, axis=1).max() for a in A]); p80, p90 = np.percentile(s[elig], [80, 90]); rng = np.random.default_rng(20261023)
core = elig[s[elig] < p80]; buf = elig[(s[elig] >= p80) & (s[elig] < p90)]; high = elig[s[elig] >= p90]
core = rng.permutation(core); nval = int(round(0.1 * len(core))); central_val, core_train = np.sort(core[:nval]), np.sort(core[nval:])
high = rng.permutation(high); n = len(high); h_dev, h_seal, h_broad = np.sort(high[: n // 2]), np.sort(high[n // 2: n // 2 + n // 4]), np.sort(high[n // 2 + n // 4:])
k_add = len(buf) + len(h_broad); broad = np.sort(np.concatenate([rng.choice(core_train, len(core_train) - k_add, replace=False), buf, h_broad]))
assert len(broad) == len(core_train)
excl_all = set(central_val) | set(h_dev) | set(h_seal) | sealed_v1
assert not (set(core_train) & (excl_all | set(buf) | set(h_broad))) and not (set(broad) & excl_all)
bound = max(np.linalg.norm(A[i], axis=1).max() for i in core_train)
# window exposure audit: start grid 0, 9, 18, 27 (official length 37; 64-frame episodes allow starts 0..27)
L = 37; win = []
for i in h_dev:
    nrm = np.linalg.norm(A[i], axis=1)
    for s0 in range(0, len(nrm) - L + 1, 9):
        w = nrm[s0:s0 + L]; ex = np.where(w > bound)[0]
        win.append({"e": int(i), "s0": int(s0), "n_exceed": int(len(ex)), "first_exceed": int(ex[0]) if len(ex) else -1, "context_precedes": bool(len(ex) and ex[0] > 0)})
clean_wins = [w for w in win if w["n_exceed"] > 0 and w["context_precedes"]]
ex_eps = {w["e"] for w in clean_wins}
# configuration audit from frame 0 (observables only)
bg = np.load(os.environ.get("BG_CLEAN", os.path.join(HERE, "bg_clean.npy"))).astype(np.float64); bg = bg.transpose(1, 2, 0) if bg.shape[0] == 3 else bg
yy, xx = np.mgrid[0:240, 0:240]


def frame0(i):
    p = os.path.join(DATA, os.path.basename(metas[i]["video_path"]))
    if not os.path.exists(p): return None
    with av.open(p) as c:
        for f in c.decode(video=0):
            a = f.to_ndarray(format="rgb24"); return (cv2.resize(a, (240, 240), interpolation=cv2.INTER_AREA) if a.shape[:2] != (240, 240) else a).astype(np.float64) / 255


def obs(i):
    Y = frame0(i)
    if Y is None: return None
    bgr = cv2.resize(bg, (240, 240)) if bg.shape[:2] != (240, 240) else bg
    m = (np.maximum(np.clip(Y - bgr, 0, None) / np.clip(1 - bgr, 1e-3, None), np.clip(bgr - Y, 0, None) / np.clip(bgr, 1e-3, None)).max(-1) > 0.5)
    a = m.sum(); return [float(a), float((m * xx).sum() / max(a, 1)), float((m * yy).sum() / max(a, 1))]


O = {g: [o for o in (obs(i) for i in idx) if o is not None] for g, idx in (("core_train", core_train), ("high_dev", h_dev))}
qs = [10, 25, 50, 75, 90]; conf = {g: {k: np.percentile(np.array(v)[:, j], qs).tolist() for j, k in enumerate(("area", "cx", "cy"))} | {"n": len(v)} for g, v in O.items()}
out = {"version": 2, "amends": os.path.basename(V1), "amends_md5": v1_hash, "reason": "v1 scored per-coordinate rarity (dominated by low-scale coordinate 5); v2 scores action amplitude max_t ||a_t||_2 (raw units, physical zero), chosen before any model exists",
       "score": "max_t ||a_t||_2", "p80": float(p80), "p90": float(p90), "core_amplitude_bound": float(bound),
       "groups": {"core_train": core_train.tolist(), "central_val": central_val.tolist(), "buffer": np.sort(buf).tolist(), "high_dev": h_dev.tolist(), "high_sealed_v2": h_seal.tolist(),
                  "high_broad": h_broad.tolist(), "broad_support_train": broad.tolist(), "sealed_v1_excluded": sorted(sealed_v1)},
       "counts": {k: int(len(v)) for k, v in (("core_train", core_train), ("central_val", central_val), ("buffer", buf), ("high_dev", h_dev), ("high_sealed_v2", h_seal), ("high_broad", h_broad), ("broad_support_train", broad), ("sealed_v1_excluded", sealed_v1))},
       "s_range": {k: [float(s[v].min()), float(s[v].max())] for k, v in (("core_train", core_train), ("buffer", buf), ("high", high))},
       "window_exposure": {"grid": "s0 in 0, 9, 18, 27", "n_high_dev_windows": len(win), "n_with_exceedance": int(sum(w["n_exceed"] > 0 for w in win)),
                           "n_clean_action_extrapolation_windows": len(clean_wins), "n_episodes_with_clean_window": len(ex_eps),
                           "first_exceed_quartiles": np.percentile([w["first_exceed"] for w in clean_wins], [25, 50, 75]).tolist() if clean_wins else None,
                           "n_exceed_quartiles": np.percentile([w["n_exceed"] for w in clean_wins], [25, 50, 75]).tolist() if clean_wins else None, "windows": win},
       "configuration_audit_frame0": conf | {"note": "local videos only (subset of episodes); observables from frame 0 vs bg_clean"},
       "note": "high_sealed_v2 and all v1-sealed episodes must not be read during development; no official ind_test / ood_test file is used; nothing here used model errors."}
json.dump(out, open(os.path.join(HERE, "clean_split_robot_arm_v2.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("counts", "s_range", "p80", "p90", "core_amplitude_bound")}, indent=1))
print({k: v for k, v in out["window_exposure"].items() if k != "windows"}); print(json.dumps(conf, indent=0)[:900])
