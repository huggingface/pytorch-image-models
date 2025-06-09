#!/usr/bin/env python3
import sys
import subprocess
import json
from pathlib import Path

best_dir = Path("/home/s175698/zasn/timm/output/opt_trial_ID6_cspresnet50_173/opt_trial_ID6_cspresnet50_173")
checkpoint = best_dir / "model_best.pth.tar"
if not checkpoint.exists():
    raise FileNotFoundError(f"Brak checkpointa pod ścieżką: {checkpoint}")

DATA_DIR    = "../data/tiny-imagenet-200"
MODEL       = "cspresnet50"
NUM_CLASSES = "200"
IMG_SIZE    = "64"
BATCH_SIZE  = "256"

def run_validate(split: str) -> float:
    cmd = [
        sys.executable, "-u", "validate.py",
        "--data-dir", DATA_DIR,
        "--model", MODEL,
        "--num-classes", NUM_CLASSES,
        "--img-size", IMG_SIZE,
        "--batch-size", BATCH_SIZE,
        "--split", split,
        "--checkpoint", str(checkpoint),
    ]
    print(f">>> Running: {' '.join(cmd)}\n")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    print(completed.stdout, completed.stderr, sep="\n")
    if completed.returncode != 0:
        raise RuntimeError(f"`validate.py` zakończyło się błędem (code {completed.returncode})")
    if "--result\n" not in completed.stdout:
        raise RuntimeError("Nie znalazłem w outpucie `--result` z JSONem.")
    _, json_part = completed.stdout.split("--result\n", 1)
    results = json.loads(json_part)
    if isinstance(results, list):
        results = results[0]
    return float(results["top1"])

if __name__ == "__main__":
    print("=== Sanity‐check na TRAINING ===")
    train_top1 = run_validate("train")
    print(f"Training Top-1 = {train_top1:.2f}%\n")

    print("=== Sanity‐check na VALIDATION ===")
    val_top1 = run_validate("validation")
    print(f"Validation Top-1 = {val_top1:.2f}%\n")

    print("=== Końcowy test na TEST ===")
    test_top1 = run_validate("test")
    print(f"Test Top-1 = {test_top1:.2f}%")
