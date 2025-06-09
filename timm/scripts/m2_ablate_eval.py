#!/usr/bin/env python3
import sys
import subprocess
import json
from pathlib import Path
import csv

dir = Path("/home/s175698/zasn/timm/output/ablate_seed_resnet50_45/ablate_seed_resnet50_45")

file = dir / "summary.csv"
if not file.exists():
    raise FileNotFoundError(f"Brak pliku pod ścieżką: {file}")

def get_top1(csv_file: Path) -> float:
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        best_eval_top1 = 0.0
        for row in reader:
            if "eval_top1" in row:
                eval_top1 = float(row["eval_top1"])
                if eval_top1 > best_eval_top1:
                    best_eval_top1 = eval_top1
        return best_eval_top1
             
    raise RuntimeError("Nie znalazłem w pliku CSV kolumny 'top1'.")

if __name__ == "__main__":
    print("=== Sanity‐check na VALIDATION ===")
    val_top1 = get_top1(file)
    print(f"Validation Top-1 = {val_top1:.2f}%\n")
