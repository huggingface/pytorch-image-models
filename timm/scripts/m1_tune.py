#!/usr/bin/env python3
import optuna
import subprocess
import re
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from optuna.exceptions import TrialPruned

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ID = 7
NAME = "ghostnet_100"
DATASET = "../data/tiny-imagenet-200"
WANDB_PROJECT = "tiny-imagenet-optuna-training"
WANDB_TAGS = "optuna-pruning-test"
OPTIMIZERS = ["sgd", "adam", "adamw", "adagrad", "rmsprop"]

_cp_re = re.compile(r"checkpoint-(\d+)\.pth\.tar[\"']?,\s*([\d.]+)")
GLOBAL_BEST = {"value": -float("inf"), "out_dir": None, "trial": None}

def clean_checkpoints(path: Path):
    for p in path.rglob("*.pth.tar"):
        if p.name != "model_best.pth.tar":
            try:
                p.unlink()
            except FileNotFoundError:
                pass

def run_train(trial, opt, lr, momentum, weight_decay, smoothing):
    exp_name = f"opt_trial_ID{ID}_{NAME}_{trial.number}"
    out_dir = Path("output") / exp_name
    cmd = [
        sys.executable, "-u", "train.py", DATASET,
        "--model", NAME,
        "--num-classes", "200",
        "--epochs", "150",
        "--batch-size", "256",
        "--img-size", "64",
        "--opt", opt,
        "--lr", str(lr),
        "--momentum", str(momentum),
        "--weight-decay", str(weight_decay),
        "--smoothing", str(smoothing),
        "--warmup-epochs", "0",
        "--output", str(out_dir),
        "--experiment", exp_name,
        "--log-wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-tags", WANDB_TAGS,
        "--checkpoint-hist", "1",
        "--recovery-interval", "0"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    best = -float("inf")
    try:
        for line in proc.stdout:
            print(line, end="")
            m = _cp_re.search(line)
            if m:
                epoch_idx = int(m.group(1))
                top1 = float(m.group(2))
                best = max(best, top1)
                trial.report(best, epoch_idx + 1)
                if trial.should_prune():
                    proc.kill()
                    raise TrialPruned()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError()
        csv = out_dir / "summary.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            best = max(best, df["eval_top1"].max())
        if best == -float("inf"):
            raise RuntimeError()
        return best, out_dir
    finally:
        if out_dir.exists():
            clean_checkpoints(out_dir)

def manage_folders(score: float, out_dir: Path, trial_number: int):
    global GLOBAL_BEST
    if score > GLOBAL_BEST["value"]:
        if GLOBAL_BEST["out_dir"] and GLOBAL_BEST["out_dir"].exists():
            shutil.rmtree(GLOBAL_BEST["out_dir"], ignore_errors=True)
        GLOBAL_BEST = {"value": score, "out_dir": out_dir, "trial": trial_number}
    else:
        shutil.rmtree(out_dir, ignore_errors=True)

def objective(trial):
    opt = trial.suggest_categorical("opt", ["sgd", "adam", "adamw", "adagrad", "rmsprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    smoothing = trial.suggest_float("smoothing", 0.0, 0.3)
    score, out_dir = run_train(trial, opt, lr, momentum, weight_decay, smoothing)
    manage_folders(score, out_dir, trial.number)
    return score

if __name__ == "__main__":
    os.makedirs("optuna_study", exist_ok=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=2)
    study = optuna.create_study(
        study_name=f"{NAME}_training",
        direction="maximize",
        pruner=pruner,
        storage=f"sqlite:///optuna_study/{NAME}_training.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=25)
    print("=== Najlepsze parametry ===")
    print(study.best_trial.params)
    print(f"Best Top-1 val: {study.best_value:.2f}%")
    print(f"Best trial number: {study.best_trial.number}")
