from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")
INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7o_rule_based_comparison"
    / "step7o_three_way_episode_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7p_risk_robustness"
)

MODES = [
    "passive_grid_only",
    "rule_based_bess_ems",
    "full_fc_hmarl",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.95)
    return p.parse_args()


def upper_tail_var_cvar(values, alpha=0.95):
    """
    For costs: larger values are worse.
    VaR_alpha = alpha quantile.
    CVaR_alpha = mean of observations >= VaR_alpha.
    """
    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size == 0:
        return np.nan, np.nan

    var = float(np.quantile(x, alpha))
    tail = x[x >= var]

    cvar = float(np.mean(tail)) if tail.size else var
    return var, cvar


def lower_tail_var_cvar_returns(values, alpha=0.95):
    """
    For returns: smaller (more negative) values are worse.

    For alpha=0.95, the adverse lower tail is the worst 5%.
    Returns:
        lower_tail_var = 5th percentile
        lower_tail_cvar = mean of returns <= that percentile
    """
    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size == 0:
        return np.nan, np.nan

    q = 1.0 - alpha
    var = float(np.quantile(x, q))
    tail = x[x <= var]

    cvar = float(np.mean(tail)) if tail.size else var
    return var, cvar


def safe_cv(mean, std):
    if abs(mean) < 1e-12:
        return np.nan
    return float(std / abs(mean))


def main():
    args = parse_args()

    if not (0.5 < args.alpha < 1.0):
        raise ValueError("--alpha must be between 0.5 and 1.0.")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required Step 7O-B file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "mode",
        "test_episode",
        "total_return",
        "net_market_cost_usd",
        "grid_import_kwh",
        "peak_grid_import_kw",
        "power_balance_violation_steps",
        "transformer_violation_steps",
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(
            "Missing required columns in Step 7O-B file: "
            + ", ".join(missing)
        )

    print("=" * 80)
    print("STEP 7P - RISK AND ROBUSTNESS COMPARISON")
    print("=" * 80)
    print("Input         :", INPUT_FILE)
    print("Risk level    :", args.alpha)
    print("TEST use      : final evaluation only")
    print("Learning      : disabled")
    print("Model tuning  : disabled")
    print()

    rows = []

    for mode in MODES:
        d = df[df["mode"] == mode].copy()

        if d.empty:
            raise ValueError(f"No rows found for mode: {mode}")

        costs = d["net_market_cost_usd"].to_numpy(dtype=float)
        returns = d["total_return"].to_numpy(dtype=float)
        imports = d["grid_import_kwh"].to_numpy(dtype=float)
        peaks = d["peak_grid_import_kw"].to_numpy(dtype=float)

        cost_var, cost_cvar = upper_tail_var_cvar(
            costs,
            alpha=args.alpha,
        )

        return_var, return_cvar = lower_tail_var_cvar_returns(
            returns,
            alpha=args.alpha,
        )

        cost_mean = float(np.mean(costs))
        cost_std = float(np.std(costs, ddof=0))

        row = {
            "mode": mode,
            "episodes": int(len(d)),

            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=0)),
            "median_return": float(np.median(returns)),
            "worst_return": float(np.min(returns)),
            "best_return": float(np.max(returns)),

            "lower_tail_return_var": return_var,
            "lower_tail_return_cvar": return_cvar,

            "mean_net_market_cost_usd": cost_mean,
            "std_net_market_cost_usd": cost_std,
            "median_net_market_cost_usd": float(np.median(costs)),
            "minimum_net_market_cost_usd": float(np.min(costs)),
            "maximum_net_market_cost_usd": float(np.max(costs)),

            "cost_var95_usd": cost_var,
            "cost_cvar95_usd": cost_cvar,
            "cost_coefficient_of_variation": safe_cv(
                cost_mean,
                cost_std,
            ),

            "mean_grid_import_kwh": float(np.mean(imports)),
            "std_grid_import_kwh": float(np.std(imports, ddof=0)),
            "maximum_grid_import_kwh": float(np.max(imports)),

            "mean_peak_grid_import_kw": float(np.mean(peaks)),
            "maximum_peak_grid_import_kw": float(np.max(peaks)),

            "power_balance_violation_steps": int(
                d["power_balance_violation_steps"].sum()
            ),
            "transformer_violation_steps": int(
                d["transformer_violation_steps"].sum()
            ),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    passive = summary[
        summary["mode"] == "passive_grid_only"
    ].iloc[0]

    rule = summary[
        summary["mode"] == "rule_based_bess_ems"
    ].iloc[0]

    fc = summary[
        summary["mode"] == "full_fc_hmarl"
    ].iloc[0]

    def reduction(ref, proposed):
        if abs(float(ref)) < 1e-12:
            return np.nan
        return 100.0 * (float(ref) - float(proposed)) / abs(float(ref))

    comparisons = {
        "alpha": args.alpha,

        "fc_vs_passive_cost_var_reduction_percent":
            reduction(
                passive["cost_var95_usd"],
                fc["cost_var95_usd"],
            ),

        "fc_vs_passive_cost_cvar_reduction_percent":
            reduction(
                passive["cost_cvar95_usd"],
                fc["cost_cvar95_usd"],
            ),

        "fc_vs_rule_cost_var_reduction_percent":
            reduction(
                rule["cost_var95_usd"],
                fc["cost_var95_usd"],
            ),

        "fc_vs_rule_cost_cvar_reduction_percent":
            reduction(
                rule["cost_cvar95_usd"],
                fc["cost_cvar95_usd"],
            ),

        "fc_vs_passive_cost_std_reduction_percent":
            reduction(
                passive["std_net_market_cost_usd"],
                fc["std_net_market_cost_usd"],
            ),

        "fc_vs_rule_cost_std_reduction_percent":
            reduction(
                rule["std_net_market_cost_usd"],
                fc["std_net_market_cost_usd"],
            ),

        "fc_vs_passive_worst_cost_reduction_percent":
            reduction(
                passive["maximum_net_market_cost_usd"],
                fc["maximum_net_market_cost_usd"],
            ),

        "fc_vs_rule_worst_cost_reduction_percent":
            reduction(
                rule["maximum_net_market_cost_usd"],
                fc["maximum_net_market_cost_usd"],
            ),

        "fc_vs_passive_worst_return_improvement":
            float(fc["worst_return"] - passive["worst_return"]),

        "fc_vs_rule_worst_return_improvement":
            float(fc["worst_return"] - rule["worst_return"]),

        "fc_vs_passive_lower_tail_return_cvar_improvement":
            float(
                fc["lower_tail_return_cvar"]
                - passive["lower_tail_return_cvar"]
            ),

        "fc_vs_rule_lower_tail_return_cvar_improvement":
            float(
                fc["lower_tail_return_cvar"]
                - rule["lower_tail_return_cvar"]
            ),
    }

    summary_file = (
        OUTPUT_DIR
        / "step7p_risk_robustness_summary.csv"
    )

    comparison_file = (
        OUTPUT_DIR
        / "step7p_risk_comparison.json"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    comparison_file.write_text(
        json.dumps(comparisons, indent=4),
        encoding="utf-8",
    )

    print("=" * 80)
    print("STEP 7P RISK SUMMARY")
    print("=" * 80)

    display_cols = [
        "mode",
        "mean_net_market_cost_usd",
        "std_net_market_cost_usd",
        "cost_var95_usd",
        "cost_cvar95_usd",
        "maximum_net_market_cost_usd",
        "mean_return",
        "worst_return",
        "lower_tail_return_cvar",
        "power_balance_violation_steps",
        "transformer_violation_steps",
    ]

    print(
        summary[display_cols].to_string(index=False)
    )

    print()
    print("FC-HMARL vs passive:")
    print(
        f"  Cost VaR reduction  : "
        f"{comparisons['fc_vs_passive_cost_var_reduction_percent']:.4f}%"
    )
    print(
        f"  Cost CVaR reduction : "
        f"{comparisons['fc_vs_passive_cost_cvar_reduction_percent']:.4f}%"
    )
    print(
        f"  Cost std reduction  : "
        f"{comparisons['fc_vs_passive_cost_std_reduction_percent']:.4f}%"
    )
    print(
        f"  Worst-cost reduction: "
        f"{comparisons['fc_vs_passive_worst_cost_reduction_percent']:.4f}%"
    )
    print(
        f"  Worst-return improvement: "
        f"{comparisons['fc_vs_passive_worst_return_improvement']:.6f}"
    )

    print()
    print("FC-HMARL vs rule-based EMS:")
    print(
        f"  Cost VaR reduction  : "
        f"{comparisons['fc_vs_rule_cost_var_reduction_percent']:.4f}%"
    )
    print(
        f"  Cost CVaR reduction : "
        f"{comparisons['fc_vs_rule_cost_cvar_reduction_percent']:.4f}%"
    )
    print(
        f"  Cost std reduction  : "
        f"{comparisons['fc_vs_rule_cost_std_reduction_percent']:.4f}%"
    )
    print(
        f"  Worst-cost reduction: "
        f"{comparisons['fc_vs_rule_worst_cost_reduction_percent']:.4f}%"
    )
    print(
        f"  Worst-return improvement: "
        f"{comparisons['fc_vs_rule_worst_return_improvement']:.6f}"
    )

    print()
    print("[OK] Step 7P risk/robustness comparison complete.")
    print("Summary file    :", summary_file)
    print("Comparison JSON :", comparison_file)


if __name__ == "__main__":
    main()
