# The question: an action-conditioned video world model is given commands larger than any it was trained on. What does it predict?


## Setup

**Two models.** We train two world models with the same architecture and the same training budget. The only difference is that one of them never sees the episodes with the largest commands.
- **Restricted model R:** trained on episodes whose largest command norm is below the 80th percentile, minus a 10% central validation set.
- **Broad model B:** the same number of episodes, including high-amplitude ones.
- **Training, both models:** DiT-S from random initialization, the released training loop, the frozen pretrained Wan2.1 VAE, 100k updates each.

**Splits** (`clean_split_v2.py`): whole episodes are assigned by their largest command norm.

| Group | Episodes | Used for |
|---|---|---|
| Below the 80th percentile | R's training data (minus central validation) | training R |
| 80th to 90th percentile | buffer | never seen by R |
| Above the 90th percentile, 50% | `high_dev` | evaluation |
| Above the 90th percentile, 25% | `high_sealed_v2` | held out, never read |
| Above the 90th percentile, 25% | `high_broad` | training B only |

**Evaluation** (`clean_eval.py`):
- The official 50-step sampler, 2 draws per window.
- 64 central windows.
- 98 out-of-range windows. In each, the context frame comes before the first command above R's range. The median overshoot is 1.15 times the largest training command norm.
- Metric: the official masked MSE (M-MSE).

## Results

All results come from one training run per model. Official M-MSE ×1e-3:

| Windows | R (restricted) | B (broad) | Log gap |
|---|---|---|---|
| Out-of-range (98) | 15.58 | 9.42 | 0.50 |
| Central (64) | 9.99 | 7.33 | 0.31 |

- **Extra out-of-range penalty.** The penalty is 0.19 in log units [0.03, 0.35], bootstrap 95% CI.
- **Pull-back.** On out-of-range windows, R pulls the arm back into configurations from its training range.
  - The share of late predicted frames that are closer to some training frame of R than to their own true frame is 0.93 for R and 0.62 for B: difference +0.31 [0.23, 0.39].
  - The true late frames of these windows are as close to the training data as central windows (median distance ratio 0.99). So the correct end states lie inside the training range, and the failure comes from the commands.
- **R is also worse inside its range.** Its M-MSE on central windows is 36% higher (9.99 vs 7.33).
- **Sampling spread does not flag the out-of-range windows.** The spread between R's two draws is 22% of its late error on out-of-range windows, against 24% on central windows. Most of the extra error is a bias shared by both draws.
- **Neither simple explanation of the excess holds.**
  - How far the commands go beyond the range does not explain the size of the excess: partial Spearman +0.10 [−0.10, 0.30].
  - How unusual the true states are does not explain it either. If anything, the excess is smaller when the true late states are farther from the training frames: −0.22 [−0.41, −0.01].

The numbers are in `results/clean_report_R_B.json` and `results/pair_analysis.json` (summary in `results/pair_analysis.txt`). Per-window values are in `results/pair_analysis_perwindow.npz`.

## Limitations

- One training run per model and one environment (the robot arm).
- Only 2 draws per window.
- The model checkpoints and generated videos are not included because of their size.

## Requirements

- The official ACWM-Phys code (the `acwm` package), the robot-arm data and the Wan2.1 VAE checkpoint.
- The scripts import `acwm` from `$W/dit_repo/repo` and read data and checkpoints under `$W` (environment variable `W`, default the current directory).
- The split scripts read the robot-arm training data from `$ROBOT_DATA`.
- The configuration audit in `clean_split_v2.py` also needs `$BG_CLEAN`: the per-pixel median of the last frame over the training episodes, saved as a (3, H, W) float array in [0, 1].

## Files

| File | Role |
|---|---|
| `clean_split.py`, `clean_split_v2.py` | Episode splits by command extremeness (v1) and command amplitude (v2, used for all results) |
| `clean_cache.py` | Pre-encodes the training windows with the frozen VAE |
| `clean_train.py` | Trains a DiT-S world model from random initialization on one split group with the released training loop |
| `clean_eval.py` | Generation with the official sampler on the central and out-of-range windows, and the out-of-range penalty with bootstrap CI |
| `pair_analysis.py` | Pull-back, error correlations and the shared-bias / seed-spread split |
| `results/clean_split_robot_arm*.json` | The episode splits (indices into the official training episodes) |
| `results/train_log_R.json`, `results/train_log_B.json` | Per-update training loss and gradient norm over the last 5,000 of the 100k updates |

## How to run

```bash
python clean_split.py
python clean_split_v2.py
SPLIT=clean_split_robot_arm_v2.json OUT=cache python clean_cache.py
SPLIT=clean_split_robot_arm_v2.json GROUP=core_train OUT=ckpt_R python clean_train.py train
SPLIT=clean_split_robot_arm_v2.json GROUP=broad_support_train OUT=ckpt_B python clean_train.py train
python clean_eval.py R ckpt_R/<checkpoint>
python clean_eval.py B ckpt_B/<checkpoint>
python clean_eval.py report R B
python pair_analysis.py
```

Every script's header documents its environment variables.
