"""Generates and scores predictions of the restricted and broad models on the evaluation windows.

Models: checkpoints written by clean_train.py (restricted = core_train, broad = broad_support_train).

Windows (fixed before any model was trained, from inputs only):
 central : 64 central_val windows, episodes sorted, seeded (rng 1312) choice of episodes and of starts on the grid {0, 9, 18, 27}
 high : the out-of-range windows of clean_split_robot_arm_v2.json (window_exposure: the context frame precedes the first command above the core
 amplitude bound), one per high_dev episode (98)

Sampler: the released generate(mode='parallel', 50 steps, noise_level 0), 2 draws per window (seed = crc32 of the window key + '#k'), raw actions.
Metrics per draw (as the official eval.py compute_metrics): MSE and official M-MSE (weight 0.01 + max_{t,c}|gt - gt_0|) over all 37 frames; also
the late block 25-36 and, for high windows, the frames after the first exceedance.

Usage: python clean_eval.py TAG CKPT -> <OUT>/clean_eval_<TAG>.npz ; python clean_eval.py report TAG_R TAG_B -> out-of-range penalty with bootstrap CI"""
import os, sys, json, zlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.environ.get("W", "."); OUT = os.environ.get("OUT", os.path.join(HERE, "clean_out"))
SPLIT = os.environ.get("SPLIT", os.path.join(HERE, "clean_split_robot_arm_v2.json")); S = json.load(open(SPLIT)); L = 37
if sys.argv[1] == "report":
    tr, tb = sys.argv[2], sys.argv[3]; Rr, Rb = np.load(os.path.join(OUT, f"clean_eval_{tr}.npz")), np.load(os.path.join(OUT, f"clean_eval_{tb}.npz"))
    def per(Z, grp, met):
        m = Z["group"] == grp; ks = sorted(set(Z["key"][m].tolist())); return np.array([Z[met][m & (Z["key"] == k)].mean() for k in ks])
    lines, out = [], {}
    for met in ("mmse", "mse", "mmse_late"):
        rH, bH, rC, bC = per(Rr, "high", met), per(Rb, "high", met), per(Rr, "central", met), per(Rb, "central", met)
        ixH = np.random.default_rng(0).integers(0, len(rH), (4000, len(rH))); ixC = np.random.default_rng(1).integers(0, len(rC), (4000, len(rC)))
        D = np.log(rH.mean() / bH.mean()) - np.log(rC.mean() / bC.mean()); Db = np.log(rH[ixH].mean(1) / bH[ixH].mean(1)) - np.log(rC[ixC].mean(1) / bC[ixC].mean(1))
        gH = np.log(rH.mean() / bH.mean()); gC = np.log(rC.mean() / bC.mean())
        out[met] = {"R_restricted_high": float(rH.mean()), "R_broad_high": float(bH.mean()), "R_restricted_central": float(rC.mean()), "R_broad_central": float(bC.mean()),
                    "log_gap_high": float(gH), "log_gap_central": float(gC), "D_coverage": float(D), "D_coverage_ci": np.percentile(Db, [2.5, 97.5]).tolist()}
        lines.append(f"{met:9s} high R {rH.mean() * 1e3:.4f} vs B {bH.mean() * 1e3:.4f} (log gap {gH:+.3f}) | central R {rC.mean() * 1e3:.4f} vs B {bC.mean() * 1e3:.4f} "
                     f"(log gap {gC:+.3f}) | D_coverage {D:+.3f} [{np.percentile(Db, 2.5):+.3f}, {np.percentile(Db, 97.5):+.3f}]")
    json.dump(out, open(os.path.join(OUT, f"clean_report_{tr}_{tb}.json"), "w"), indent=1); print("\n".join(lines)); sys.exit()
TAG, CKPT = sys.argv[1], sys.argv[2]
import torch, yaml, cv2, av
DEV = "cuda" if torch.cuda.is_available() else "cpu"; REPO = os.path.join(W, "dit_repo", "repo"); VAE = os.environ.get("VAE", os.path.join(W, "dit_ckpt", "Wan2.1_VAE.pth"))
DATA = os.environ.get("ROBOT_DATA", os.path.join(W, "acwm/data/kinematics/robot_arm_64/ind_train"))
sys.path.insert(0, REPO); cwd = os.getcwd(); os.chdir(REPO)
from acwm.dataset.data_config import get_config_by_name
from acwm.model.interface import get_dynamics_class
cfg = yaml.safe_load(open("configs/envs/robot_arm.yaml")); cfg["model_config"]["action_dim"] = get_config_by_name(cfg["dataset"]["name"]).action_dim
cfg["model_config"].update(yaml.safe_load(open(f"configs/model/{cfg['model_type']}.yaml")).get("model_config", {})); cfg["model_config"]["vae_config"] = [VAE]
host = get_dynamics_class(cfg["dynamics_class"])(cfg["model_name"], cfg["model_config"]); os.chdir(cwd)
ck = torch.load(CKPT, map_location="cpu", weights_only=False); res = host.load_state_dict(ck["model_state_dict"], strict=False)
assert not [k for k in res.missing_keys if not k.startswith("vae.")], res.missing_keys[:5]
host.to(DEV).eval(); metas = torch.load(os.path.join(DATA, "metadata.pt"), weights_only=False); G = S["groups"]
never = set(G["high_sealed_v2"]) | set(G["sealed_v1_excluded"])
rng = np.random.default_rng(1312); cen_eps = sorted(G["central_val"]); pick = rng.choice(len(cen_eps), 64, replace=False)
wins = [("central", int(cen_eps[j]), int(rng.choice([0, 9, 18, 27]))) for j in sorted(pick)]
wins += [("high", int(w["e"]), int(w["s0"])) for w in S["window_exposure"]["windows"] if w["n_exceed"] > 0 and w["context_precedes"]]
assert not ({w[1] for w in wins} & never) and not ({w[1] for w in wins} & (set(G[ck.get("group", "core_train")]) if ck.get("group") in G else set()))
first_ex = {(w["e"], w["s0"]): w["first_exceed"] for w in S["window_exposure"]["windows"]}
NFE = int(os.environ.get("NFE", 50)); MAXWIN = int(os.environ.get("MAXWIN", 0))
if MAXWIN: wins = wins[:MAXWIN // 2] + [w for w in wins if w[0] == "high"][:MAXWIN - MAXWIN // 2]


def clip(ti, s0):
    out = []
    with av.open(os.path.join(DATA, os.path.basename(metas[ti]["video_path"]))) as c:
        for st in c.streams.video: st.thread_count = 1
        for f in c.decode(video=0):
            a = f.to_ndarray(format="rgb24"); out.append(cv2.resize(a, (240, 240), interpolation=cv2.INTER_AREA) if a.shape[:2] != (240, 240) else a)
    v = np.stack(out).astype(np.float32) / 255; gt = v[s0:s0 + L]
    if len(gt) < L: gt = np.concatenate([gt, np.repeat(gt[-1:], L - len(gt), 0)])
    A = np.asarray(metas[ti]["actions"], np.float32)[s0:s0 + L]
    if len(A) < L: A = np.concatenate([A, np.repeat(A[-1:], L - len(A), 0)])
    return gt, torch.from_numpy(A)[None]


rows = {k: [] for k in ("group", "key", "draw", "mse", "mmse", "mmse_late", "mmse_post")}; P = {}; EB = int(os.environ.get("EB", 8))
for k in (1, 2):
    for bi in range(0, len(wins), EB):
        Wk = wins[bi:bi + EB]; C = [clip(w[1], w[2]) for w in Wk]; seed = zlib.crc32((repr(Wk) + f"#{k}").encode()); torch.manual_seed(seed)
        if DEV == "cuda": torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            H = host.generate(torch.from_numpy(np.stack([c[0][0] for c in C])).to(DEV), torch.cat([c[1] for c in C]).to(DEV), num_inference_steps=NFE, noise_level=0.0, mode="parallel").float().cpu()
        for j, w in enumerate(Wk):
            g = torch.from_numpy(C[j][0]); pr = H[j, :L].clamp(0, 1); wt = (0.01 + (g - g[:1]).abs().amax(dim=-1, keepdim=True).amax(dim=0, keepdim=True)).expand_as(g)
            mm = lambda lo, hi: float((wt[lo:hi] * (pr[lo:hi] - g[lo:hi]) ** 2).sum() / wt[lo:hi].sum())
            fe = first_ex.get((w[1], w[2]), 0) if w[0] == "high" else 0
            rows["group"].append(w[0]); rows["key"].append(f"{w[1]}_{w[2]}"); rows["draw"].append(k); rows["mse"].append(float(((pr - g) ** 2).mean()))
            rows["mmse"].append(mm(0, L)); rows["mmse_late"].append(mm(25, L)); rows["mmse_post"].append(mm(max(fe, 1), L))
            P[(w[1], w[2], k)] = (pr[1:, ::2, ::2] * 255 + 0.5).byte().numpy()
        print("CLEAN_EVAL", TAG, "draw", k, bi + len(Wk), len(wins), flush=True)
keys = [(w[1], w[2]) for w in wins]
np.savez(os.path.join(OUT, f"clean_eval_{TAG}.npz"), **{k: np.array(v) for k, v in rows.items()}, D1=np.stack([P[(e, s, 1)] for e, s in keys]), D2=np.stack([P[(e, s, 2)] for e, s in keys]),
         win_group=np.array([w[0] for w in wins]), win_e=np.array([w[1] for w in wins]), win_s0=np.array([w[2] for w in wins]), ckpt_step=ck.get("step", -1))
print("CLEAN_EVAL_DONE", TAG, len(rows["key"]), flush=True)
