"""
STEP 7R-Q1 — BUILD FINAL BENCHMARK + DIAGNOSTIC ABLATION PUBLICATION TABLES

Reads ONLY already-generated FINAL V3 evaluation results.
No controller evaluation, training, replay write, or checkpoint selection.

Outputs:
1) Table_Final_Benchmark_Comparison.csv/.md/.txt
2) Table_Final_Diagnostic_Ablation.csv/.md/.txt
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

BENCH_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_benchmark"
ABL_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_ablation"
TEST_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_test"

BENCH_SUMMARY = BENCH_DIR / "final_v3_benchmark_summary.csv"
BENCH_PROTOCOL = BENCH_DIR / "final_v3_benchmark_protocol.json"

ABL_SUMMARY = ABL_DIR / "final_v3_ablation_summary.csv"
ABL_PAIRED = ABL_DIR / "final_v3_ablation_paired_comparisons.csv"
ABL_METADATA = ABL_DIR / "final_v3_ablation_metadata.json"

LOCKED_TEST = TEST_DIR / "final_test_summary.csv"

OUT_DIR = TEST_DIR / "publication_tables"


def section(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def pct_reduction(reference, proposed):
    reference = float(reference)
    proposed = float(proposed)
    if abs(reference) < 1e-12:
        return np.nan
    return 100.0 * (reference - proposed) / abs(reference)


def write_table(df, stem, title):
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    txt_path = OUT_DIR / f"{stem}.txt"

    df.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(
        df.to_markdown(index=False, floatfmt=".6f"),
        encoding="utf-8",
    )
    txt_path.write_text(
        title + "\n\n"
        + df.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
        + "\n",
        encoding="utf-8",
    )

    return csv_path, md_path, txt_path


def main():
    section("STEP 7R-Q1 — FINAL PUBLICATION TABLES")

    for p in (
        BENCH_SUMMARY,
        BENCH_PROTOCOL,
        ABL_SUMMARY,
        ABL_PAIRED,
        ABL_METADATA,
        LOCKED_TEST,
    ):
        require(p)

    bench = pd.read_csv(BENCH_SUMMARY)
    ablation = pd.read_csv(ABL_SUMMARY)
    paired = pd.read_csv(ABL_PAIRED)
    locked = pd.read_csv(LOCKED_TEST).iloc[0]

    bench_protocol = json.loads(BENCH_PROTOCOL.read_text(encoding="utf-8"))
    abl_metadata = json.loads(ABL_METADATA.read_text(encoding="utf-8"))

    # ---------------------------------------------------------
    # PROTOCOL AUDIT
    # ---------------------------------------------------------
    if int(bench_protocol["checkpoint"]) != 200:
        raise RuntimeError("Benchmark is not locked to checkpoint 200.")
    if int(bench_protocol["test_windows"]) != 1268:
        raise RuntimeError("Benchmark did not use all 1268 TEST windows.")
    if not bool(bench_protocol["deterministic_fc_hmarl"]):
        raise RuntimeError("Benchmark FC-HMARL evaluation was not deterministic.")
    if not bool(bench_protocol["learning_disabled"]):
        raise RuntimeError("Benchmark learning was not disabled.")

    if int(abl_metadata["checkpoint"]) != 200:
        raise RuntimeError("Ablation is not locked to checkpoint 200.")
    if int(abl_metadata["test_windows"]) != 1268:
        raise RuntimeError("Ablation did not use all 1268 TEST windows.")
    if abl_metadata["ablation_type"] != "post_training_inference_diagnostic":
        raise RuntimeError("Unexpected ablation type.")

    locked_mean = float(locked["mean_return"])

    bench_full = bench.loc[
        bench["mode"] == "full_fc_hmarl",
        "mean_return",
    ].iloc[0]

    abl_full = ablation.loc[
        ablation["mode"] == "full_fc_hmarl",
        "mean_return",
    ].iloc[0]

    if abs(float(bench_full) - locked_mean) > 1e-6:
        raise RuntimeError("Benchmark full mode does not reproduce locked TEST mean.")
    if abs(float(abl_full) - locked_mean) > 1e-6:
        raise RuntimeError("Ablation full mode does not reproduce locked TEST mean.")

    # ---------------------------------------------------------
    # TABLE 1 — BENCHMARK
    # ---------------------------------------------------------
    labels = {
        "passive_grid_only": "Passive grid-only",
        "rule_based_bess_ems": "Rule-based BESS EMS",
        "full_fc_hmarl": "FC-HMARL",
    }

    benchmark_rows = []

    for mode in (
        "passive_grid_only",
        "rule_based_bess_ems",
        "full_fc_hmarl",
    ):
        r = bench[bench["mode"] == mode].iloc[0]

        benchmark_rows.append({
            "Controller": labels[mode],
            "Mean return": float(r["mean_return"]),
            "Return std.": float(r["std_return"]),
            "Net market cost (USD/day)": float(r["mean_net_market_cost_usd"]),
            "Grid import (kWh/day)": float(r["mean_grid_import_kwh"]),
            "Grid export (kWh/day)": float(r["mean_grid_export_kwh"]),
            "Peak grid import (kW)": float(r["mean_peak_grid_import_kw"]),
            "BESS throughput (kWh/day)": float(r["mean_bess_throughput_kwh"]),
            "Energy sharing (kWh/day)": float(r["mean_sharing_scheduled_kwh"]),
            "Feasible reserve (kWh/day)": float(r["mean_reserve_feasible_kwh"]),
            "Balance violation steps": int(r["power_balance_violation_steps"]),
            "Transformer violation steps": int(r["transformer_violation_steps"]),
            "SOC violation events": int(r["soc_below_min_count"] + r["soc_above_max_count"]),
        })

    benchmark_table = pd.DataFrame(benchmark_rows)

    passive = bench[bench["mode"] == "passive_grid_only"].iloc[0]
    rule = bench[bench["mode"] == "rule_based_bess_ems"].iloc[0]
    full = bench[bench["mode"] == "full_fc_hmarl"].iloc[0]

    improvement_table = pd.DataFrame([
        {
            "Comparison": "FC-HMARL vs passive grid-only",
            "Return improvement": float(full["mean_return"] - passive["mean_return"]),
            "Market-cost reduction (%)": pct_reduction(
                passive["mean_net_market_cost_usd"],
                full["mean_net_market_cost_usd"],
            ),
            "Grid-import reduction (%)": pct_reduction(
                passive["mean_grid_import_kwh"],
                full["mean_grid_import_kwh"],
            ),
            "Peak-import reduction (%)": pct_reduction(
                passive["mean_peak_grid_import_kw"],
                full["mean_peak_grid_import_kw"],
            ),
        },
        {
            "Comparison": "FC-HMARL vs rule-based BESS EMS",
            "Return improvement": float(full["mean_return"] - rule["mean_return"]),
            "Market-cost reduction (%)": pct_reduction(
                rule["mean_net_market_cost_usd"],
                full["mean_net_market_cost_usd"],
            ),
            "Grid-import reduction (%)": pct_reduction(
                rule["mean_grid_import_kwh"],
                full["mean_grid_import_kwh"],
            ),
            "Peak-import reduction (%)": pct_reduction(
                rule["mean_peak_grid_import_kw"],
                full["mean_peak_grid_import_kw"],
            ),
        },
    ])

    # ---------------------------------------------------------
    # TABLE 2 — DIAGNOSTIC ABLATION
    # ---------------------------------------------------------
    abl_labels = {
        "full_fc_hmarl": "Full FC-HMARL",
        "no_confidence_awareness": "Without confidence awareness",
        "no_energy_sharing": "Without energy sharing",
        "no_upper_level_coordination": "Without upper-level coordination",
    }

    paired_lookup = {
        str(r["comparison"]): r
        for _, r in paired.iterrows()
    }

    ablation_rows = []

    full_abl = ablation[ablation["mode"] == "full_fc_hmarl"].iloc[0]

    for mode in (
        "full_fc_hmarl",
        "no_confidence_awareness",
        "no_energy_sharing",
        "no_upper_level_coordination",
    ):
        r = ablation[ablation["mode"] == mode].iloc[0]

        if mode == "full_fc_hmarl":
            delta = 0.0
            alt_better = 0
            full_better = 0
        else:
            key = f"{mode}_minus_full"
            pr = paired_lookup[key]
            delta = float(pr["mean_delta_return"])
            alt_better = int(pr["alternative_better_windows"])
            full_better = int(pr["full_better_windows"])

        ablation_rows.append({
            "Diagnostic configuration": abl_labels[mode],
            "Mean return": float(r["mean_return"]),
            "Return change vs full": delta,
            "Net market cost (USD/day)": float(r["mean_net_market_cost_usd"]),
            "Grid import (kWh/day)": float(r["mean_grid_import_kwh"]),
            "Peak grid import (kW)": float(r["mean_peak_grid_import_kw"]),
            "Energy sharing (kWh/day)": float(r["mean_sharing_scheduled_kwh"]),
            "Feasible reserve (kWh/day)": float(r["mean_reserve_feasible_kwh"]),
            "Alternative-better windows": alt_better,
            "Full-better windows": full_better,
            "Balance violation steps": int(r["power_balance_violation_steps"]),
            "Transformer violation steps": int(r["transformer_violation_steps"]),
            "SOC violation events": int(r["soc_below_min_count"] + r["soc_above_max_count"]),
        })

    ablation_table = pd.DataFrame(ablation_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bench_files = write_table(
        benchmark_table,
        "Table_Final_Benchmark_Comparison",
        "Table — Final benchmark comparison on 1,268 locked TEST windows",
    )

    improve_files = write_table(
        improvement_table,
        "Table_Final_Benchmark_Improvements",
        "Table — FC-HMARL improvements relative to benchmark controllers",
    )

    abl_files = write_table(
        ablation_table,
        "Table_Final_Diagnostic_Ablation",
        "Table — Post-training inference-time diagnostic ablation",
    )

    section("TABLE 1 — FINAL BENCHMARK COMPARISON")
    print(
        benchmark_table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    section("TABLE 1B — FC-HMARL IMPROVEMENTS")
    print(
        improvement_table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    section("TABLE 2 — DIAGNOSTIC ABLATION")
    print(
        ablation_table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    section("REPORTING INTERPRETATION")
    print(
        "1. FC-HMARL outperforms both benchmark controllers in mean return "
        "and mean net market cost."
    )
    print(
        "2. FC-HMARL does NOT reduce grid import relative to the rule-based EMS; "
        "report that metric exactly rather than claiming universal improvement."
    )
    print(
        "3. Removing upper-level coordination causes a substantial performance "
        "degradation and introduces balance-violation steps."
    )
    print(
        "4. Removing confidence awareness or energy sharing gives a very small "
        "increase in TEST return in this inference-time intervention."
    )
    print(
        "5. Therefore, do NOT claim that every component individually improves "
        "TEST return."
    )
    print(
        "6. The ablation results are post-training inference diagnostics, NOT "
        "separately retrained ablation controllers."
    )

    section("STEP 7R-Q1 COMPLETE")
    print("Benchmark table:")
    for p in bench_files:
        print(p)
    print()
    print("Benchmark improvement table:")
    for p in improve_files:
        print(p)
    print()
    print("Diagnostic ablation table:")
    for p in abl_files:
        print(p)

    print()
    print("[OK] Locked checkpoint 200 verified.")
    print("[OK] 1,268 TEST windows verified.")
    print("[OK] Benchmark and ablation full modes reproduce locked TEST mean.")
    print("[OK] No policy evaluation, training, or checkpoint selection occurred.")


if __name__ == "__main__":
    main()
