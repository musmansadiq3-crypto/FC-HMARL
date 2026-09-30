from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_comment3_ablation.py"

VARIANTS = [
    "full",
    "no_confidence",
    "state_only",
    "reward_only",
    "concatenated",
    "multiplicative",
    "no_coordinator",
]

SEEDS = [11, 22, 33, 44, 55]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FC-HMARL Comment-3 multi-seed ablation runner."
    )

    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        default="all",
    )

    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    variants = (
        VARIANTS
        if args.variant == "all"
        else [args.variant]
    )

    seeds = (
        SEEDS
        if args.seed is None
        else [args.seed]
    )

    output_root = ROOT / "outputs" / "comment3_ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = output_root / "comment3_run_manifest.csv"
    rows = []

    print("=" * 72)
    print("COMMENT 3 - MULTI-SEED ABLATION")
    print("=" * 72)
    print("Variants :", variants)
    print("Seeds    :", seeds)
    print("Episodes :", args.episodes)

    for variant in variants:
        for seed in seeds:

            print()
            print("=" * 72)
            print(f"VARIANT={variant} | SEED={seed}")
            print("=" * 72)

            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--episodes",
                str(args.episodes),
                "--seed",
                str(seed),
                "--ablation-variant",
                variant,
            ]

            if args.smoke:
                command.append("--smoke-test")

            result = subprocess.run(
                command,
                cwd=str(ROOT),
            )

            rows.append({
                "variant": variant,
                "seed": seed,
                "episodes": args.episodes,
                "return_code": result.returncode,
            })

            with manifest.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant",
                        "seed",
                        "episodes",
                        "return_code",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            if result.returncode != 0:
                print(f"[FAILED] {variant}, seed={seed}")
                raise SystemExit(result.returncode)

            print(f"[OK] {variant}, seed={seed}")

    print()
    print("=" * 72)
    print("COMMENT 3 RUNNER COMPLETE")
    print("=" * 72)
    print("Completed runs:", len(rows))
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
