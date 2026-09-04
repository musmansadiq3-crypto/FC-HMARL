from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecasting.model import (
    ForecastModelConfig,
    MultiHorizonTransformerForecaster,
)


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_real_forecasting_model.pt"
)

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_sequences.npz"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_scaler.csv"
)

EXISTING_TEST_ARCHIVE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "real_forecasting_test_predictions.npz"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "best_checkpoint_verification"
)

REGENERATED_ARCHIVE = (
    OUTPUT_DIRECTORY
    / "best_checkpoint_test_predictions_regenerated.npz"
)

REPORT_JSON = (
    OUTPUT_DIRECTORY
    / "best_checkpoint_test_prediction_verification.json"
)

REPORT_CSV = (
    OUTPUT_DIRECTORY
    / "best_checkpoint_test_prediction_verification.csv"
)

BATCH_SIZE = 128
DEVICE = "cpu"


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inverse_transform(values, minimums, ranges):
    values = np.asarray(values, dtype=np.float32)
    return (
        values * ranges[None, None, :]
        + minimums[None, None, :]
    ).astype(np.float32)


def generate_predictions(model, x_test, batch_size, device):
    predictions = []

    model.eval()

    with torch.no_grad():

        for start in range(0, len(x_test), batch_size):

            end = min(
                start + batch_size,
                len(x_test),
            )

            batch = torch.as_tensor(
                x_test[start:end],
                dtype=torch.float32,
                device=device,
            )

            output = model(batch)

            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "Expected Transformer forward() to return "
                    "a torch.Tensor."
                )

            predictions.append(
                output.detach().cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0,
    ).astype(np.float32)


def compare_arrays(name, regenerated, existing):
    regenerated = np.asarray(regenerated)
    existing = np.asarray(existing)

    if regenerated.shape != existing.shape:
        return {
            "array": name,
            "shape_match": False,
            "regenerated_shape": str(regenerated.shape),
            "existing_shape": str(existing.shape),
            "maximum_absolute_difference": None,
            "mean_absolute_difference": None,
            "rmse_difference": None,
            "allclose_atol_1e-6": False,
            "allclose_atol_1e-5": False,
        }

    diff = (
        regenerated.astype(np.float64)
        - existing.astype(np.float64)
    )

    abs_diff = np.abs(diff)

    return {
        "array": name,
        "shape_match": True,
        "regenerated_shape": str(regenerated.shape),
        "existing_shape": str(existing.shape),
        "maximum_absolute_difference":
            float(np.max(abs_diff)),
        "mean_absolute_difference":
            float(np.mean(abs_diff)),
        "rmse_difference":
            float(np.sqrt(np.mean(diff ** 2))),
        "allclose_atol_1e-6":
            bool(np.allclose(
                regenerated,
                existing,
                rtol=0.0,
                atol=1e-6,
            )),
        "allclose_atol_1e-5":
            bool(np.allclose(
                regenerated,
                existing,
                rtol=0.0,
                atol=1e-5,
            )),
    }


def main():

    section(
        "STEP 7L-E - VERIFY TEST TRANSFORMER "
        "PREDICTION PROVENANCE"
    )

    for path in (
        CHECKPOINT_FILE,
        SEQUENCE_FILE,
        SCALER_FILE,
        EXISTING_TEST_ARCHIVE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD BEST VALIDATION CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_config" not in checkpoint:
        raise RuntimeError(
            "model_config not found in checkpoint."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "model_state_dict not found in checkpoint."
        )

    config = ForecastModelConfig(
        **checkpoint["model_config"]
    )

    model = MultiHorizonTransformerForecaster(
        config
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best checkpoint epoch       : "
        f"{checkpoint.get('epoch')}"
    )
    print(
        f"Best validation loss        : "
        f"{checkpoint.get('validation_loss')}"
    )
    print(
        f"Model input window          : "
        f"{config.input_window}"
    )
    print(
        f"Model forecast horizon      : "
        f"{config.forecast_horizon}"
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    sequences = np.load(
        SEQUENCE_FILE,
        allow_pickle=True,
    )

    x_test = np.asarray(
        sequences["X_test"],
        dtype=np.float32,
    )

    y_test = np.asarray(
        sequences["y_test"],
        dtype=np.float32,
    )

    feature_names = np.asarray(
        sequences["feature_names"]
    )

    print(
        f"X_test shape                : "
        f"{x_test.shape}"
    )
    print(
        f"y_test shape                : "
        f"{y_test.shape}"
    )

    # ========================================================
    # REGENERATE TEST PREDICTIONS FROM BEST CHECKPOINT
    # ========================================================

    section(
        "REGENERATING TEST PREDICTIONS"
    )

    regenerated_normalized = (
        generate_predictions(
            model=model,
            x_test=x_test,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )
    )

    if (
        regenerated_normalized.shape
        != y_test.shape
    ):
        raise RuntimeError(
            "Regenerated prediction shape does not "
            "match y_test."
        )

    scaler_df = pd.read_csv(
        SCALER_FILE
    )

    minimums = (
        scaler_df["train_min"]
        .to_numpy(dtype=np.float32)
    )

    ranges = (
        scaler_df["train_range"]
        .to_numpy(dtype=np.float32)
    )

    regenerated_original = inverse_transform(
        regenerated_normalized,
        minimums,
        ranges,
    )

    targets_original = inverse_transform(
        y_test,
        minimums,
        ranges,
    )

    np.savez_compressed(
        REGENERATED_ARCHIVE,
        predictions_normalized=
            regenerated_normalized,
        targets_normalized=
            y_test,
        predictions_original=
            regenerated_original,
        targets_original=
            targets_original,
        feature_names=
            feature_names,
    )

    print(
        "Regenerated archive saved : "
        f"{REGENERATED_ARCHIVE}"
    )

    # ========================================================
    # LOAD EXISTING TEST ARCHIVE
    # ========================================================

    existing = np.load(
        EXISTING_TEST_ARCHIVE,
        allow_pickle=True,
    )

    expected_keys = [
        "predictions_normalized",
        "targets_normalized",
        "predictions_original",
        "targets_original",
        "feature_names",
    ]

    print()
    print(
        "Existing archive keys      : "
        f"{existing.files}"
    )

    missing = [
        key for key in expected_keys
        if key not in existing.files
    ]

    if missing:
        raise RuntimeError(
            "Existing TEST archive is missing keys: "
            f"{missing}"
        )

    # ========================================================
    # ELEMENT-BY-ELEMENT COMPARISON
    # ========================================================

    section(
        "COMPARING REGENERATED VS EXISTING TEST ARCHIVE"
    )

    comparisons = []

    comparisons.append(
        compare_arrays(
            "predictions_normalized",
            regenerated_normalized,
            existing["predictions_normalized"],
        )
    )

    comparisons.append(
        compare_arrays(
            "targets_normalized",
            y_test,
            existing["targets_normalized"],
        )
    )

    comparisons.append(
        compare_arrays(
            "predictions_original",
            regenerated_original,
            existing["predictions_original"],
        )
    )

    comparisons.append(
        compare_arrays(
            "targets_original",
            targets_original,
            existing["targets_original"],
        )
    )

    comparison_df = pd.DataFrame(
        comparisons
    )

    comparison_df.to_csv(
        REPORT_CSV,
        index=False,
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    feature_names_match = bool(
        np.array_equal(
            feature_names,
            existing["feature_names"],
        )
    )

    normalized_prediction_match_1e6 = bool(
        comparisons[0][
            "allclose_atol_1e-6"
        ]
    )

    normalized_prediction_match_1e5 = bool(
        comparisons[0][
            "allclose_atol_1e-5"
        ]
    )

    original_prediction_match_1e6 = bool(
        comparisons[2][
            "allclose_atol_1e-6"
        ]
    )

    original_prediction_match_1e5 = bool(
        comparisons[2][
            "allclose_atol_1e-5"
        ]
    )

    strict_match = (
        normalized_prediction_match_1e6
        and original_prediction_match_1e6
        and feature_names_match
    )

    practical_match = (
        normalized_prediction_match_1e5
        and original_prediction_match_1e5
        and feature_names_match
    )

    report = {
        "best_checkpoint_file":
            str(CHECKPOINT_FILE),
        "best_checkpoint_epoch":
            checkpoint.get("epoch"),
        "best_checkpoint_validation_loss":
            checkpoint.get("validation_loss"),
        "existing_test_archive":
            str(EXISTING_TEST_ARCHIVE),
        "regenerated_test_archive":
            str(REGENERATED_ARCHIVE),
        "test_samples":
            int(len(x_test)),
        "forecast_horizon":
            int(config.forecast_horizon),
        "target_features":
            int(config.target_features),
        "feature_names_match":
            feature_names_match,
        "strict_match_atol_1e-6":
            strict_match,
        "practical_match_atol_1e-5":
            practical_match,
        "comparisons":
            comparisons,
    }

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    section(
        "VERIFICATION VERDICT"
    )

    print(
        f"Feature names match         : "
        f"{feature_names_match}"
    )

    print(
        f"Strict prediction match     : "
        f"{strict_match}"
    )

    print(
        f"Practical prediction match  : "
        f"{practical_match}"
    )

    print(
        "Maximum normalized pred diff: "
        f"{comparisons[0]['maximum_absolute_difference']}"
    )

    print(
        "Maximum original pred diff  : "
        f"{comparisons[2]['maximum_absolute_difference']}"
    )

    if strict_match:
        print()
        print(
            "[OK] Existing TEST predictions are reproduced "
            "from best_real_forecasting_model.pt."
        )
        print(
            "[OK] The existing TEST Transformer archive has "
            "verified best-checkpoint provenance."
        )

    elif practical_match:
        print()
        print(
            "[OK] Existing TEST predictions match the best "
            "checkpoint within 1e-5 numerical tolerance."
        )
        print(
            "[NOTE] Small floating-point differences are present."
        )

    else:
        print()
        print(
            "[WARNING] Existing TEST predictions do NOT match "
            "the best saved forecasting checkpoint."
        )
        print(
            "[ACTION] Do not overwrite the old archive yet. "
            "Use the regenerated archive for a controlled "
            "comparison before finalizing FC-HMARL TEST results."
        )

    print()
    print(
        f"Verification CSV:\n{REPORT_CSV}"
    )
    print()
    print(
        f"Verification JSON:\n{REPORT_JSON}"
    )


if __name__ == "__main__":
    main()
