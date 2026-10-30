"""Episode split by command extremeness (version 1). CPU, metadata only (actions); no video and no official test file.
Robot-arm ind_train episodes (metadata.pt). Episode score s_e = max_{t, d} |(a_{e,t,d} - m_d) / r_d| with a robust centre m_d (median over all
actions of all episodes) and scale r_d (IQR / 1.349), fixed before any model exists. Whole episodes inherit one assignment:
 core s_e < p80 -> restricted-model training, except a seeded central validation subset (10% of core, used for checkpoint selection)
 excluded p80 <= s_e < p90 -> never used by the restricted model (a buffer between core and the high region)
 high s_e >= p90 -> seeded 50 / 25 / 25 split into high_dev (evaluation), high_sealed (a held-out confirmation set that is never read)
 and high_reserved (training pool for the broad-support model)
Also writes a same-size training list for the broad-support model, and reports per-group counts, score ranges, per-dimension action ranges and
how far high_dev actions lie beyond the core-train range. clean_split_v2.py (the split used for all results) keeps this file's held-out set.
Usage: python clean_split.py -> clean_split_robot_arm.json"""
import os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ROBOT_DATA", os.path.join(HERE, "acwm", "data", "kinematics", "robot_arm_64", "ind_train"))
metas = torch.load(os.path.join(DATA, "metadata.pt"), weights_only=False); A = [np.asarray(m["actions"], np.float64) for m in metas]
allA = np.concatenate(A); m = np.median(allA, 0); q75, q25 = np.percentile(allA, [75, 25], 0); r = np.maximum((q75 - q25) / 1.349, 1e-9)
s = np.array([np.abs((a - m) / r).max() for a in A]); p80, p90 = np.percentile(s, [80, 90]); rng = np.random.default_rng(2026)
core = np.where(s < p80)[0]; excl = np.where((s >= p80) & (s < p90))[0]; high = np.where(s >= p90)[0]
core = rng.permutation(core); nval = int(round(0.1 * len(core))); central_val, core_train = np.sort(core[:nval]), np.sort(core[nval:])
high = rng.permutation(high); n = len(high); h_dev, h_seal, h_res = np.sort(high[: n // 2]), np.sort(high[n // 2: n // 2 + n // 4]), np.sort(high[n // 2 + n // 4:])
broad = np.sort(np.concatenate([rng.choice(core_train, len(core_train) - len(h_res), replace=False), h_res]))
ctA = np.concatenate([A[i] for i in core_train]); lo, hi = ctA.min(0), ctA.max(0); hdA = np.concatenate([A[i] for i in h_dev])
outside = ((hdA < lo) | (hdA > hi)).mean(0)
out = {"n_episodes": len(metas), "centre": m.tolist(), "scale": r.tolist(), "p80": float(p80), "p90": float(p90),
       "groups": {"core_train": core_train.tolist(), "central_val": central_val.tolist(), "excluded": np.sort(excl).tolist(), "high_dev": h_dev.tolist(),
                  "high_sealed": h_seal.tolist(), "high_reserved": h_res.tolist(), "broad_support_train": broad.tolist()},
       "counts": {k: int(len(v)) for k, v in (("core_train", core_train), ("central_val", central_val), ("excluded", excl), ("high_dev", h_dev),
                                              ("high_sealed", h_seal), ("high_reserved", h_res), ("broad_support_train", broad))},
       "s_range": {k: [float(s[v].min()), float(s[v].max())] for k, v in (("core", core), ("excluded", excl), ("high", high))},
       "core_train_action_range": [lo.tolist(), hi.tolist()], "high_dev_frac_outside_core_range_per_dim": outside.tolist(),
       "high_dev_frac_any_dim_outside_per_frame": float(((hdA < lo) | (hdA > hi)).any(1).mean()),
       "note": "high_sealed is written for the record only and must not be read during development; no official ind_test / ood_test file is used."}
assert not (set(core_train) & set(h_dev)) and not (set(broad) & (set(h_dev) | set(h_seal) | set(central_val)))
json.dump(out, open(os.path.join(HERE, "clean_split_robot_arm.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("counts", "s_range", "p80", "p90")}, indent=1)); print("high_dev fraction outside core range per dim", np.round(outside, 3), "any dim per frame", round(out["high_dev_frac_any_dim_outside_per_frame"], 3))
