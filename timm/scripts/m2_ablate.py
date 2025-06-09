#!/usr/bin/env python3
import subprocess, sys, json, re, csv, statistics, shutil
from pathlib import Path

MODEL          = "resnet50"
BEST_PARAMS    = dict(
    opt          = "adamw",
    lr           = 0.0006884108823417583,
    momentum     = 0.8346985153547885,
    weight_decay = 0.00013280955139120207,
    smoothing    = 0.2587194356028499,
)
DATASET        = "../data/tiny-imagenet-200"
EPOCHS         = "150"
BATCH_SIZE     = "256"
IMG_SIZE       = "64"
CUDA_DEV       = "3"         

SEEDS = [45]         
BASE_OUT = Path("output")

# -----------------------------------------------------------------------------
def run(seed: int):
    out_dir = BASE_OUT / f"ablate_seed_{MODEL}_{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable, "-u", "train.py", DATASET,
        "--model", MODEL,
        #"--model-kwargs", "use_ghost=False",        
        "--num-classes", "200",
        "--epochs", EPOCHS,
        "--batch-size", BATCH_SIZE,
        "--img-size", IMG_SIZE,
        "--opt", BEST_PARAMS["opt"],
        "--lr", str(BEST_PARAMS["lr"]),
        "--momentum", str(BEST_PARAMS["momentum"]),
        "--weight-decay", str(BEST_PARAMS["weight_decay"]),
        "--smoothing", str(BEST_PARAMS["smoothing"]),
        "--warmup-epochs", "0",
        "--output", str(out_dir),
        "--experiment", f"ablate_seed_{MODEL}_{seed}",
        "--checkpoint-hist", "1",
        "--recovery-interval", "0",
        "--seed", str(seed),
        "--log-wandb",
        "--wandb-project", "tiny-imagenet-resnet50-abl",
        "--wandb-tags", "ablation",
    ]
    env = dict(**os.environ, CUDA_VISIBLE_DEVICES=CUDA_DEV)
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

def read_top1(csv_file: Path):
    with csv_file.open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["eval_top1"]), float(rows[-1]["eval_top5"])

if __name__ == "__main__":
    import os
    BASE_OUT.mkdir(exist_ok=True)

    for s in SEEDS:
        print(f"\n=== Trening z seed = {s} ===")
        run(s)

    results = []
    for s in SEEDS:
        summ = BASE_OUT / f"ablate_seed_{MODEL}_{s}/ablate_seed_{MODEL}_{s}/summary.csv"
        top1, top5 = read_top1(summ)
        results.append((s, top1, top5))

    print("\n=== PODSUMOWANIE ABLACJI resnet50 ===")
    print(f"{'seed':>6} | {'Top-1':>6} | {'Top-5':>6}")
    print("-"*24)
    for s, t1, t5 in results:
        print(f"{s:>6} | {t1:6.2f} | {t5:6.2f}")
    print("-"*24)

    with open("ablate_results_{MODEL}_{SEED}.json", "w") as f:
        json.dump({
            "runs": [{"seed": s, "top1": t1, "top5": t5} for s, t1, t5 in results],
        }, f, indent=4)
    print("\n--> Wyniki zapisane w ablate_results.json")
