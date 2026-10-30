"""Usage: SPLIT=... ACWM_DATA_ROOT=... OUT=... SHARD=0 NSHARD=2 python clean_cache.py"""
import os, sys, json, time, numpy as np, torch, yaml
W = os.environ.get("W", "."); REPO = os.path.join(W, "dit_repo", "repo"); VAE = os.environ.get("VAE", os.path.join(W, "dit_ckpt", "Wan2.1_VAE.pth"))
SPLIT = os.environ["SPLIT"]; OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True); SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
STARTS = [int(x) for x in os.environ.get("STARTS", ",".join(str(s) for s in range(0, 28, 3))).split(",")]; EB = int(os.environ.get("EB", 8)); DEV = "cuda" if torch.cuda.is_available() else "cpu"; MAXW = int(os.environ.get("MAXW", 0))
sys.path.insert(0, REPO); cwd = os.getcwd(); os.chdir(REPO)
from acwm.dataset.data_config import get_config_by_name
from acwm.model.interface import get_dynamics_class
from acwm.dataset.dataset import RoboticsDatasetWrapper
cfg = yaml.safe_load(open("configs/envs/robot_arm.yaml")); cfg["model_config"]["action_dim"] = get_config_by_name(cfg["dataset"]["name"]).action_dim
cfg["model_config"].update(yaml.safe_load(open(f"configs/model/{cfg['model_type']}.yaml")).get("model_config", {})); cfg["model_config"]["vae_config"] = [VAE]
host = get_dynamics_class(cfg["dynamics_class"])(cfg["model_name"], cfg["model_config"]).to(DEV).eval()
dk = cfg["dataset"].copy(); dk.pop("name", None); dk["seq_len"] = dk.get("train_seq_len", dk.get("seq_len"))
ds = RoboticsDatasetWrapper.get_dataset(cfg["dataset"]["name"], split="ind_train", max_trajs=None, **dk); os.chdir(cwd)
S = json.load(open(SPLIT)); G = S["groups"]; use = set(G["core_train"]) | set(G["broad_support_train"]) | set(G["central_val"])
never = set(G["high_dev"]) | set(G["high_sealed_v2"]) | set(G["sealed_v1_excluded"]); assert not (use & never)
sel = sorted((i, s) for (i, s) in ds.indices if i in use and s in STARTS); sel = sel[SH::NSH]; sel = sel[:MAXW] if MAXW else sel; ds.indices = sel
print(f"CACHE shard {SH}/{NSH}: {len(sel)} windows from {len({i for i, _ in sel})} episodes; starts {STARTS}", flush=True)
loader = torch.utils.data.DataLoader(ds, batch_size=EB, shuffle=False, num_workers=int(os.environ.get("WORKERS", 6)), drop_last=False)
N = len(sel); lat = np.lib.format.open_memmap(os.path.join(OUT, f"lat_{SH}.npy"), mode="w+", dtype=np.float16, shape=(N, 10, 30, 30, 16))
acts = None; k = 0; t0 = time.time()
with torch.no_grad():
    for b in loader:
        z = host.encode_obs(b["obs"].to(DEV)).float(); assert tuple(z.shape[1:]) == (10, 30, 30, 16), z.shape
        lat[k:k + len(z)] = z.cpu().numpy().astype(np.float16); a = b["action"].float().numpy()
        if acts is None: acts = np.zeros((N,) + a.shape[1:], np.float32)
        acts[k:k + len(z)] = a; k += len(z)
        if (k // EB) % 100 == 0: print("CACHE", SH, k, N, f"{(time.time() - t0) / k:.3f}s/window", flush=True)
lat.flush(); np.save(os.path.join(OUT, f"act_{SH}.npy"), acts); np.save(os.path.join(OUT, f"idx_{SH}.npy"), np.array(sel, np.int64))
print("CACHE_DONE", SH, k, round(time.time() - t0), "s", flush=True)
