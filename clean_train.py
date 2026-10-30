"""Trains a DiT-S robot world model from random initialisation on one split group (measurement model, not a method).

Released DiffusionForcing_WM class with the robot_arm / dit_s configs, frozen pretrained Wan2.1 VAE, trained ONLY on the episodes of one group of
the split file (default core_train). Windows come from the official ind_train dataset (all sliding starts, official preprocessing), filtered to the
allowed episodes. Each update is exactly the released loop: encode_obs (no grad) -> training_loss -> clip_grad_norm 1.0 -> AdamW lr 1e-4 (no schedule,
no EMA, as released). Optional gradient accumulation (ACCUM). No ind_test / ood_test loader exists in this file.

Validation: fixed central_val noised tuples (seeded windows, levels and noise) -> mean training loss every VAL_FREQ updates; checkpoints every
CKPT_FREQ. Generation-based checks are done separately with clean_eval.py.
Modes: timing (WARM warm-up + N timed updates; prints s/update and a 100k-update cost table) | train.

Env: W (root, default .), SPLIT (json), GROUP (core_train), OUT, BATCH (4), ACCUM (1), STEPS (100000), N (200), WARM (20),
 VAL_N (64), VAL_FREQ (2000), CKPT_FREQ (5000), WORKERS (8), SEED (0).

Usage: python clean_train.py timing|train"""
import os, sys, json, time, math, random
MODE = sys.argv[1]; assert MODE in ("timing", "train")
import numpy as np, torch, yaml
W = os.environ.get("W", "."); REPO = os.path.join(W, "dit_repo", "repo"); VAE = os.environ.get("VAE", os.path.join(W, "dit_ckpt", "Wan2.1_VAE.pth"))
HERE = os.path.dirname(os.path.abspath(__file__)); SPLIT = os.environ.get("SPLIT", os.path.join(HERE, "clean_split_robot_arm.json")); GROUP = os.environ.get("GROUP", "core_train")
OUT = os.environ.get("OUT", os.path.join(HERE, "clean_out")); os.makedirs(OUT, exist_ok=True)
BATCH, ACCUM, STEPS, N, WARM = (int(os.environ.get(k, d)) for k, d in (("BATCH", 4), ("ACCUM", 1), ("STEPS", 100000), ("N", 200), ("WARM", 20)))
VAL_N, VAL_FREQ, CKPT_FREQ, WORKERS, SEED = (int(os.environ.get(k, d)) for k, d in (("VAL_N", 64), ("VAL_FREQ", 2000), ("CKPT_FREQ", 5000), ("WORKERS", 8), ("SEED", 0)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
CACHE = os.environ.get("CACHE", ""); BF16 = os.environ.get("BF16", "0") == "1"   # CACHE: dir of clean_cache.py shards (latents pre-encoded, fp32 VAE)
sys.path.insert(0, REPO); cwd = os.getcwd(); os.chdir(REPO)
from acwm.dataset.data_config import get_config_by_name
from acwm.model.interface import get_dynamics_class
from acwm.dataset.dataset import RoboticsDatasetWrapper
cfg = yaml.safe_load(open("configs/envs/robot_arm.yaml")); cfg["model_config"]["action_dim"] = get_config_by_name(cfg["dataset"]["name"]).action_dim
cfg["model_config"].update(yaml.safe_load(open(f"configs/model/{cfg['model_type']}.yaml")).get("model_config", {})); cfg["model_config"]["vae_config"] = [VAE]
host = get_dynamics_class(cfg["dynamics_class"])(cfg["model_name"], cfg["model_config"]).to(DEV)   # random DiT init; the VAE loads its pretrained weights
dk = cfg["dataset"].copy(); dk.pop("name", None); dk["seq_len"] = dk.get("train_seq_len", dk.get("seq_len"))
ds = None if CACHE else RoboticsDatasetWrapper.get_dataset(cfg["dataset"]["name"], split="ind_train", max_trajs=None, **dk); os.chdir(cwd)
S = json.load(open(SPLIT)); allowed = set(S["groups"][GROUP]); val_eps = set(S["groups"]["central_val"])
if not CACHE:
    nep = len({i for i, _ in ds.indices}); assert nep == S["n_episodes"] and max(i for i, _ in ds.indices) == nep - 1, (nep, S["n_episodes"])   # metadata.pt order, no max_trajs
G_ = S["groups"]; never = [k for k in ("high_dev", "high_sealed", "high_sealed_v2", "sealed_v1_excluded", "central_val") if k in G_]
restricted_only = [k for k in ("excluded", "buffer", "high_reserved", "high_broad") if k in G_] if GROUP == "core_train" else []
for k in never + restricted_only: assert not (allowed & set(G_[k])), k   # v1 and v2 manifests
if CACHE:   # latents pre-encoded by clean_cache.py (official dataset windows, fp32 VAE); no video decoding / VAE in the loop
    shards = sorted(f[4:-4] for f in os.listdir(CACHE) if f.startswith("idx_") and f.endswith(".npy"))
    LAT = [np.load(os.path.join(CACHE, f"lat_{k}.npy"), mmap_mode="r") for k in shards]; ACT = [np.load(os.path.join(CACHE, f"act_{k}.npy")) for k in shards]
    IDX = [np.load(os.path.join(CACHE, f"idx_{k}.npy")) for k in shards]
    rows = [(j, r) for j in range(len(shards)) for r in range(len(IDX[j]))]
    tr_rows = [x for x in rows if int(IDX[x[0]][x[1], 0]) in allowed]; vrows = [x for x in rows if int(IDX[x[0]][x[1], 0]) in val_eps]
    assert tr_rows and (os.environ.get("CACHE_PARTIAL_OK") == "1" or len({int(IDX[j][r, 0]) for j, r in tr_rows}) == len(allowed)), ("cache does not cover the group", len({int(IDX[j][r, 0]) for j, r in tr_rows}), len(allowed))
    print(f"CLEAN {GROUP} (cache): {len(allowed)} episodes, {len(tr_rows)} cached training windows; central_val cached windows {len(vrows)}; device {DEV}; bf16 {BF16}", flush=True)
    gsamp = np.random.default_rng(SEED + 7)
    def fetch(sel):
        z = torch.from_numpy(np.stack([np.asarray(LAT[j][r], np.float32) for j, r in sel])).to(DEV); a = torch.from_numpy(np.stack([ACT[j][r] for j, r in sel])).to(DEV); return z, a
    def next_batch():
        return fetch([tr_rows[k] for k in gsamp.integers(0, len(tr_rows), BATCH)])
    rv = random.Random(1234); vsel_rows = rv.sample(vrows, min(VAL_N, len(vrows)))
    VALZ = [fetch(vsel_rows[j:j + BATCH]) for j in range(0, len(vsel_rows), BATCH)]
    assert VALZ or os.environ.get("CACHE_PARTIAL_OK") == "1", "no cached central_val windows"
else:
    if MODE == "timing" and os.environ.get("TIMING_SUBSET") == "1":   # timing only: restrict to episodes whose video is present (data subset)
        import torch as _t; _m = _t.load(os.path.join(ds.effective_root, "metadata.pt"), weights_only=False)
        allowed = {i for i in allowed if os.path.exists(os.path.join(ds.effective_root, os.path.basename(_m[i]["video_path"])))}; del _m
        assert allowed, "no local videos for timing"
    full = list(ds.indices); ds.indices = [(i, s0) for (i, s0) in full if i in allowed]; vidx = [(i, s0) for (i, s0) in full if i in val_eps]
    print(f"CLEAN {GROUP}: {len(allowed)} episodes, {len(ds.indices)} training windows; central_val windows {len(vidx)}; device {DEV}", flush=True)
for p in host.vae.parameters(): p.requires_grad_(False)
params = [p for p in host.model.parameters()]; nparam = sum(p.numel() for p in params); print(f"DiT parameters {nparam / 1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"])); clip = float(cfg["training"]["grad_clip"])
if not CACHE:
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=(DEV == "cuda"), drop_last=True, persistent_workers=WORKERS > 0)
    # fixed validation tuples: VAL_N seeded central_val windows; levels / noise drawn from a fixed generator at every evaluation
    rv = random.Random(1234); vsel = rv.sample(vidx, min(VAL_N, len(vidx))); vsave = ds.indices
    ds.indices = vsel; VAL = [ds[j] for j in range(len(vsel))]; ds.indices = vsave
    VALZ = None


def val_loss():
    host.eval(); g = torch.Generator(device="cpu").manual_seed(4321); tot = 0.0; cnt = 0
    with torch.no_grad():
        batches_ = VALZ if CACHE else [(None, VAL[j:j + BATCH]) for j in range(0, len(VAL), BATCH)]
        for z, b in batches_:
            if not CACHE:
                obs = torch.stack([x["obs"] for x in b]).to(DEV); act = torch.stack([x["action"] for x in b]).to(DEV); z = host.encode_obs(obs)
            else:
                act = b
            st = torch.get_rng_state(); stc = torch.cuda.get_rng_state_all() if DEV == "cuda" else None; sd_ = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))
            torch.manual_seed(sd_)   # seeds CPU and CUDA generators: identical levels / noise at every evaluation
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16 and DEV == "cuda"): l_ = float(host.training_loss(z, act))
            tot += l_ * len(z); cnt += len(z); torch.set_rng_state(st)
            if stc is not None: torch.cuda.set_rng_state_all(stc)
    host.train(); return tot / max(cnt, 1)


def batches():
    while True:
        for b in loader: yield b


it = None if CACHE else batches(); host.train(); log = []; t0 = None; step = 0; total = WARM + N if MODE == "timing" else STEPS
while step < total:
    if MODE == "timing" and step == WARM:
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.time()
    opt.zero_grad(set_to_none=True); lsum = 0.0
    for _ in range(ACCUM):
        if CACHE:
            z, act = next_batch()
        else:
            b = next(it); obs, act = b["obs"].to(DEV, non_blocking=True), b["action"].to(DEV, non_blocking=True)
            with torch.no_grad(): z = host.encode_obs(obs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16 and DEV == "cuda"): loss = host.training_loss(z, act) / ACCUM
        loss.backward(); lsum += float(loss)
    gn = float(torch.nn.utils.clip_grad_norm_(params, clip)); opt.step(); step += 1; log.append((step, lsum, gn))
    if not math.isfinite(lsum): print("CLEAN_NAN", step, flush=True); sys.exit(3)
    if step % 50 == 0 or step == 1: print("STEP", step, round(float(np.mean([l for _, l, _ in log[-50:]])), 5), "gn", round(gn, 3), flush=True)
    if MODE == "train" and (step % VAL_FREQ == 0 or step == total): print("VAL", step, round(val_loss(), 6), flush=True)
    if MODE == "train" and (step % CKPT_FREQ == 0 or step == total):
        torch.save({"model_state_dict": host.state_dict(), "step": step, "group": GROUP, "split": os.path.basename(SPLIT), "batch": BATCH, "accum": ACCUM, "bf16": BF16, "cache": bool(CACHE)}, os.path.join(OUT, "latest.pt"))
        json.dump({"log": log[-5000:]}, open(os.path.join(OUT, "train_log.json"), "w")); print("CKPT", step, flush=True)
if MODE == "timing":
    if DEV == "cuda": torch.cuda.synchronize()
    sp = (time.time() - t0) / N; h = 100000 * sp / 3600
    print(f"TIMING batch {BATCH} x accum {ACCUM} (global {BATCH * ACCUM}): {sp:.3f} s/update -> 100k updates {h:.1f} h, ${0.30 * h:.2f} at $0.30/h; "
          f"peak GPU mem {torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == 'cuda' else 0:.1f} GB", flush=True)
print("CLEAN_END", MODE, flush=True)
