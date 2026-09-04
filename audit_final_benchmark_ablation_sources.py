"""
STEP 7R-P0 — AUDIT OLD BENCHMARK / ABLATION SCRIPTS

Purpose
-------
Audit the user's CURRENT local comparison scripts before any new benchmark
results are generated.

This script DOES NOT evaluate any controller and DOES NOT modify any source file.

Final protocol that must be respected
-------------------------------------
- FINAL V3 policy checkpoint: 200
- TEST windows: all 1268 valid rolling 24-h windows
- Final TEST archive: outputs/rl_data/real_rl_test_archive.npz
- Final environment / reserve / PCC corrections must be inherited from
  train_real_fc_hmarl_final_v3.py
- deterministic evaluation only
- no training, gradient update, replay write, checkpoint ranking, or re-selection
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TARGETS = [
    ROOT / "evaluate_step7o_rule_based_comparison.py",
    ROOT / "evaluate_fc_hmarl_ablation_final.py",
]

FINAL_V3 = ROOT / "train_real_fc_hmarl_final_v3.py"
FINAL_TEST_EVALUATOR = ROOT / "evaluate_final_v3_test_locked200.py"
FINAL_TEST_ARCHIVE = (
    ROOT / "outputs" / "rl_data" / "real_rl_test_archive.npz"
)

LOCKED_CHECKPOINT = 200
EXPECTED_TEST_WINDOWS = 1268


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def occurrences(text: str, pattern: str):
    return [
        m.start()
        for m in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    ]


def line_hits(text: str, pattern: str):
    rx = re.compile(
        pattern,
        flags=re.IGNORECASE,
    )
    hits = []
    for i, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if rx.search(line):
            hits.append(
                (i, line.rstrip())
            )
    return hits


def show_hits(label, hits, limit=12):
    print(f"\n{label}:")
    if not hits:
        print("  NONE")
        return
    for line_no, line in hits[:limit]:
        print(
            f"  L{line_no}: {line}"
        )
    if len(hits) > limit:
        print(
            f"  ... {len(hits) - limit} more"
        )


def audit_one(path: Path):
    section(
        f"AUDIT: {path.name}"
    )

    if not path.exists():
        print(
            "[MISSING] File is not present in project root."
        )
        return {
            "file": path.name,
            "exists": False,
            "safe_to_run": False,
            "reasons": [
                "file missing"
            ],
        }

    text = read_text(path)

    checkpoint_1000 = line_hits(
        text,
        r"\b1000\b",
    )

    checkpoint_200 = line_hits(
        text,
        r"\b200\b",
    )

    thirty_episode = line_hits(
        text,
        r"(test[-_ ]episodes.*30|default\s*=\s*30|same 30 held-out)",
    )

    old_test_import = line_hits(
        text,
        r"evaluate_real_fc_hmarl_test",
    )

    final_v3_import = line_hits(
        text,
        r"train_real_fc_hmarl_final_v3",
    )

    final_locked_eval_import = line_hits(
        text,
        r"evaluate_final_v3_test_locked200",
    )

    current_test_archive = line_hits(
        text,
        r"real_rl_test_archive\.npz",
    )

    fixed_random_starts = line_hits(
        text,
        r"build_fixed_test_episode_starts",
    )

    gradient_terms = line_hits(
        text,
        r"\.(update|train)\s*\(|backward\s*\(|optimizer\.step\s*\(",
    )

    replay_write_terms = line_hits(
        text,
        r"(replay.*(add|push|store)|store_transition)",
    )

    checkpoint_ranking_terms = line_hits(
        text,
        r"(rank.*checkpoint|best.*checkpoint|select.*checkpoint)",
    )

    corrected_reserve_refs = line_hits(
        text,
        r"(reserve_feasible|reserve_curtailed|reserve_requested)",
    )

    pcc_refs = line_hits(
        text,
        r"(requested_grid_power_kw|transformer_active_power_limit_kw|pcc)",
    )

    show_hits(
        "Checkpoint-1000 references",
        checkpoint_1000,
    )

    show_hits(
        "Checkpoint-200 references",
        checkpoint_200,
    )

    show_hits(
        "30-episode references",
        thirty_episode,
    )

    show_hits(
        "Old evaluate_real_fc_hmarl_test imports/references",
        old_test_import,
    )

    show_hits(
        "FINAL V3 imports/references",
        final_v3_import,
    )

    show_hits(
        "Locked final TEST evaluator references",
        final_locked_eval_import,
    )

    show_hits(
        "Current TEST archive references",
        current_test_archive,
    )

    show_hits(
        "Random/fixed TEST-start builder references",
        fixed_random_starts,
    )

    show_hits(
        "Gradient/training-like calls",
        gradient_terms,
    )

    show_hits(
        "Replay-write-like calls",
        replay_write_terms,
    )

    show_hits(
        "Checkpoint-ranking/selection-like calls",
        checkpoint_ranking_terms,
    )

    show_hits(
        "Corrected reserve metric references",
        corrected_reserve_refs,
    )

    show_hits(
        "PCC / requested-grid references",
        pcc_refs,
    )

    reasons = []

    if checkpoint_1000:
        reasons.append(
            "contains stale checkpoint-1000 logic"
        )

    if not checkpoint_200:
        reasons.append(
            "does not explicitly lock checkpoint 200"
        )

    if thirty_episode:
        reasons.append(
            "uses/mentions only 30 TEST episodes instead of all 1268 windows"
        )

    if old_test_import:
        reasons.append(
            "depends on old evaluate_real_fc_hmarl_test pipeline"
        )

    if not (
        final_v3_import
        or final_locked_eval_import
        or current_test_archive
    ):
        reasons.append(
            "does not visibly anchor itself to FINAL V3 / locked TEST artifacts"
        )

    if gradient_terms:
        reasons.append(
            "contains training/update-like calls that require manual inspection"
        )

    if replay_write_terms:
        reasons.append(
            "contains replay-write-like calls that require manual inspection"
        )

    safe = len(reasons) == 0

    print()
    print("-" * 92)

    if safe:
        print(
            "[PASS] No obvious stale-protocol blocker detected."
        )
    else:
        print(
            "[DO NOT RUN FOR PUBLICATION] "
            "This script is stale relative to the FINAL V3 protocol."
        )
        print()
        print("Reasons:")
        for reason in reasons:
            print(
                f"  - {reason}"
            )

    return {
        "file": path.name,
        "exists": True,
        "safe_to_run": safe,
        "reasons": reasons,
    }


def main():

    section(
        "STEP 7R-P0 — FINAL BENCHMARK / ABLATION SOURCE AUDIT"
    )

    print(
        f"Project root             : {ROOT}"
    )
    print(
        f"Required checkpoint      : {LOCKED_CHECKPOINT}"
    )
    print(
        f"Required TEST windows    : {EXPECTED_TEST_WINDOWS}"
    )
    print(
        f"FINAL V3 launcher exists : {FINAL_V3.exists()}"
    )
    print(
        f"Locked TEST evaluator    : {FINAL_TEST_EVALUATOR.exists()}"
    )
    print(
        f"Locked TEST archive      : {FINAL_TEST_ARCHIVE.exists()}"
    )

    results = [
        audit_one(path)
        for path in TARGETS
    ]

    section(
        "STEP 7R-P0 AUDIT SUMMARY"
    )

    for result in results:
        status = (
            "PASS"
            if result["safe_to_run"]
            else "STALE / DO NOT RUN"
        )

        print(
            f"{result['file']:45s} : {status}"
        )

    stale = [
        result
        for result in results
        if not result["safe_to_run"]
    ]

    print()

    if stale:
        print(
            "[EXPECTED] One or more older benchmark scripts are stale."
        )
        print(
            "Do NOT use their historical outputs in the manuscript."
        )
        print(
            "Next step: generate a new FINAL-V3-compatible benchmark "
            "evaluator locked to checkpoint 200 and all 1268 TEST windows."
        )
    else:
        print(
            "[OK] No stale blocker detected by the automatic audit."
        )
        print(
            "A manual semantic audit is still required before publication use."
        )

    print()
    print(
        "[OK] No source file was modified."
    )
    print(
        "[OK] No TEST evaluation was executed."
    )
    print(
        "[OK] Checkpoint 200 remains locked."
    )


if __name__ == "__main__":
    main()
