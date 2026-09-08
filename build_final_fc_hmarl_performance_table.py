from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3_test"
)
SUMMARY_CSV = (
    RESULT_DIR
    / "final_test_summary.csv"
)

PROTOCOL_JSON = (
    RESULT_DIR
    / "final_test_protocol.json"
)

OUTPUT_DIR = (
    RESULT_DIR
    / "publication_tables"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "Table_Final_FC_HMARL_Performance.csv"
)

OUTPUT_MD = (
    OUTPUT_DIR
    / "Table_Final_FC_HMARL_Performance.md"
)

OUTPUT_TXT = (
    OUTPUT_DIR
    / "Table_Final_FC_HMARL_Performance.txt"
)


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def format_value(value, decimals=3):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def main():

    section(
        "STEP 7R-O1 — FINAL FC-HMARL PERFORMANCE TABLE"
    )

    require_file(
        SUMMARY_CSV
    )

    require_file(
        PROTOCOL_JSON
    )

    summary_df = pd.read_csv(
        SUMMARY_CSV
    )

    if len(summary_df) != 1:
        raise RuntimeError(
            "Expected exactly one row in final_test_summary.csv."
        )

    summary = summary_df.iloc[0].to_dict()

    with open(
        PROTOCOL_JSON,
        "r",
        encoding="utf-8",
    ) as file:
        protocol = json.load(file)

    locked_checkpoint = int(
        protocol[
            "locked_checkpoint_episode"
        ]
    )

    if locked_checkpoint != 200:
        raise RuntimeError(
            "Expected validation-locked checkpoint 200."
        )

    if bool(
        protocol.get(
            "checkpoint_ranking_on_test",
            True,
        )
    ):
        raise RuntimeError(
            "TEST protocol incorrectly allows checkpoint ranking."
        )

    if bool(
        protocol.get(
            "checkpoint_reselection_after_test",
            True,
        )
    ):
        raise RuntimeError(
            "TEST protocol incorrectly allows checkpoint re-selection."
        )

    section(
        "LOCKED TEST PROTOCOL VERIFIED"
    )

    print(
        f"Locked checkpoint        : {locked_checkpoint}"
    )

    print(
        "Checkpoint selected on   : VALIDATION"
    )

    print(
        "Final evaluation split   : TEST"
    )

    print(
        "Deterministic evaluation : True"
    )

    print(
        "Checkpoint re-selection  : False"
    )

    rows = [
        {
            "Category":
                "Evaluation protocol",
            "Performance index":
                "Locked checkpoint",
            "Final TEST result":
                str(locked_checkpoint),
            "Unit":
                "episode",
        },
        {
            "Category":
                "Evaluation protocol",
            "Performance index":
                "TEST windows evaluated",
            "Final TEST result":
                str(
                    int(
                        summary[
                            "test_episodes"
                        ]
                    )
                ),
            "Unit":
                "24-h windows",
        },
        {
            "Category":
                "Learning performance",
            "Performance index":
                "Mean TEST return",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_return"
                    ],
                    6,
                ),
            "Unit":
                "reward",
        },
        {
            "Category":
                "Learning performance",
            "Performance index":
                "TEST return standard deviation",
            "Final TEST result":
                format_value(
                    summary[
                        "std_return"
                    ],
                    6,
                ),
            "Unit":
                "reward",
        },
        {
            "Category":
                "Learning performance",
            "Performance index":
                "Median TEST return",
            "Final TEST result":
                format_value(
                    summary[
                        "median_return"
                    ],
                    6,
                ),
            "Unit":
                "reward",
        },
        {
            "Category":
                "Economic",
            "Performance index":
                "Mean net market cost",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_net_market_cost_usd"
                    ],
                    3,
                ),
            "Unit":
                "USD/day",
        },
        {
            "Category":
                "Grid interaction",
            "Performance index":
                "Mean grid import",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_grid_import_kwh"
                    ],
                    3,
                ),
            "Unit":
                "kWh/day",
        },
        {
            "Category":
                "Grid interaction",
            "Performance index":
                "Mean grid export",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_grid_export_kwh"
                    ],
                    3,
                ),
            "Unit":
                "kWh/day",
        },
        {
            "Category":
                "BESS operation",
            "Performance index":
                "Mean BESS throughput",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_bess_throughput_kwh"
                    ],
                    3,
                ),
            "Unit":
                "kWh/day",
        },
        {
            "Category":
                "Energy sharing",
            "Performance index":
                "Mean scheduled energy sharing",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_sharing_scheduled_kwh"
                    ],
                    3,
                ),
            "Unit":
                "kWh/day",
        },
        {
            "Category":
                "Physical feasibility",
            "Performance index":
                "Mean power-balance violation",
            "Final TEST result":
                format_value(
                    summary[
                        "mean_mean_balance_violation_kw_per_step"
                    ],
                    9,
                ),
            "Unit":
                "kW/step",
        },
        {
            "Category":
                "Physical feasibility",
            "Performance index":
                "Transformer violation steps",
            "Final TEST result":
                str(
                    int(
                        summary[
                            "total_transformer_violation_steps"
                        ]
                    )
                ),
            "Unit":
                "steps",
        },
        {
            "Category":
                "Physical feasibility",
            "Performance index":
                "SOC lower-bound violations",
            "Final TEST result":
                str(
                    int(
                        summary[
                            "total_soc_below_min_count"
                        ]
                    )
                ),
            "Unit":
                "events",
        },
        {
            "Category":
                "Physical feasibility",
            "Performance index":
                "SOC upper-bound violations",
            "Final TEST result":
                str(
                    int(
                        summary[
                            "total_soc_above_max_count"
                        ]
                    )
                ),
            "Unit":
                "events",
        },
    ]

    table = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    markdown = table.to_markdown(
        index=False,
    )

    OUTPUT_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    lines = []
    lines.append(
        "Table — Final performance of the validation-selected FC-HMARL controller on the unseen TEST split"
    )
    lines.append(
        ""
    )

    for row in rows:
        lines.append(
            f"{row['Category']} | "
            f"{row['Performance index']} | "
            f"{row['Final TEST result']} | "
            f"{row['Unit']}"
        )

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    section(
        "PUBLICATION TABLE"
    )

    print(
        table.to_string(
            index=False
        )
    )

    section(
        "STEP 7R-O1 COMPLETE"
    )

    print(
        "CSV:"
    )
    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Markdown:"
    )
    print(
        OUTPUT_MD
    )

    print()
    print(
        "Text:"
    )
    print(
        OUTPUT_TXT
    )

    print()
    print(
        "[OK] Values were read directly from the locked FINAL TEST summary."
    )
    print(
        "[OK] No policy evaluation, training, or checkpoint selection was performed."
    )
    print(
        "[OK] Checkpoint 200 remains permanently locked."
    )


if __name__ == "__main__":
    main()
