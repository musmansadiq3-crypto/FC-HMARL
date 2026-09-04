"""
Tests for forecasting/dataset.py.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.dataset import (
    ForecastDatasetConfig,
    ForecastWindowDataset,
    build_forecasting_datasets,
    calculate_number_of_samples,
    create_forecasting_windows,
    create_window_timestamps,
    resolve_columns,
    validate_time_series,
)


# ============================================================
# TEST DATA
# ============================================================

def make_data(
    periods=300,
):

    index = pd.date_range(
        "2024-01-01",
        periods=periods,
        freq="h",
    )

    return pd.DataFrame(
        {
            "pv": np.arange(
                periods,
                dtype=float,
            ),

            "load": (
                np.arange(
                    periods,
                    dtype=float,
                )
                + 1000.0
            ),

            "ev": (
                np.arange(
                    periods,
                    dtype=float,
                )
                + 2000.0
            ),

            "price": (
                np.arange(
                    periods,
                    dtype=float,
                )
                + 3000.0
            ),
        },
        index=index,
    )


# ============================================================
# CONFIGURATION
# ============================================================

def test_default_input_window():

    config = ForecastDatasetConfig()

    assert config.input_window == 168


def test_default_forecast_horizon():

    config = ForecastDatasetConfig()

    assert config.forecast_horizon == 24


def test_default_stride():

    config = ForecastDatasetConfig()

    assert config.stride == 1


def test_valid_config():

    config = ForecastDatasetConfig()

    config.validate()


def test_invalid_input_window():

    config = ForecastDatasetConfig(
        input_window=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_forecast_horizon():

    config = ForecastDatasetConfig(
        forecast_horizon=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_stride():

    config = ForecastDatasetConfig(
        stride=0
    )

    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# TIME-SERIES VALIDATION
# ============================================================

def test_validate_time_series():

    data = make_data()

    result = validate_time_series(
        data
    )

    assert len(result) == 300


def test_empty_dataframe_rejected():

    with pytest.raises(ValueError):

        validate_time_series(
            pd.DataFrame()
        )


def test_missing_values_rejected():

    data = make_data()

    data.iloc[
        10,
        0
    ] = np.nan

    with pytest.raises(ValueError):

        validate_time_series(
            data
        )


def test_non_numeric_column_rejected():

    data = make_data()

    data[
        "label"
    ] = "example"

    with pytest.raises(TypeError):

        validate_time_series(
            data
        )


def test_unsorted_datetime_rejected():

    data = make_data()

    data = data.iloc[
        ::-1
    ]

    with pytest.raises(ValueError):

        validate_time_series(
            data
        )


# ============================================================
# COLUMN SELECTION
# ============================================================

def test_resolve_all_columns():

    data = make_data()

    columns = resolve_columns(
        data,
        None,
    )

    assert columns == [
        "pv",
        "load",
        "ev",
        "price",
    ]


def test_resolve_selected_columns():

    data = make_data()

    columns = resolve_columns(
        data,
        [
            "pv",
            "load",
        ],
    )

    assert columns == [
        "pv",
        "load",
    ]


def test_missing_selected_column_rejected():

    data = make_data()

    with pytest.raises(KeyError):

        resolve_columns(
            data,
            [
                "pv",
                "wind",
            ],
        )


# ============================================================
# SAMPLE COUNT
# ============================================================

def test_exact_minimum_length_produces_one_sample():

    number = calculate_number_of_samples(
        sequence_length=192,
        input_window=168,
        forecast_horizon=24,
    )

    assert number == 1


def test_300_hours_sample_count():

    number = calculate_number_of_samples(
        sequence_length=300,
        input_window=168,
        forecast_horizon=24,
    )

    assert number == 109


def test_short_sequence_produces_zero_samples():

    number = calculate_number_of_samples(
        sequence_length=191,
        input_window=168,
        forecast_horizon=24,
    )

    assert number == 0


def test_stride_two_sample_count():

    number = calculate_number_of_samples(
        sequence_length=300,
        input_window=168,
        forecast_horizon=24,
        stride=2,
    )

    assert number == 55


# ============================================================
# WINDOW CONSTRUCTION
# ============================================================

def test_window_shapes():

    data = make_data(
        300
    )

    X, y = create_forecasting_windows(
        data
    )

    assert X.shape == (
        109,
        168,
        4,
    )

    assert y.shape == (
        109,
        24,
        4,
    )


def test_first_input_window_values():

    data = make_data(
        300
    )

    X, _ = create_forecasting_windows(
        data
    )

    assert X[
        0,
        0,
        0
    ] == pytest.approx(
        0.0
    )

    assert X[
        0,
        -1,
        0
    ] == pytest.approx(
        167.0
    )


def test_first_target_begins_after_input():

    data = make_data(
        300
    )

    _, y = create_forecasting_windows(
        data
    )

    assert y[
        0,
        0,
        0
    ] == pytest.approx(
        168.0
    )

    assert y[
        0,
        -1,
        0
    ] == pytest.approx(
        191.0
    )


def test_second_window_moves_one_hour():

    data = make_data(
        300
    )

    X, y = create_forecasting_windows(
        data
    )

    assert X[
        1,
        0,
        0
    ] == pytest.approx(
        1.0
    )

    assert X[
        1,
        -1,
        0
    ] == pytest.approx(
        168.0
    )

    assert y[
        1,
        0,
        0
    ] == pytest.approx(
        169.0
    )


def test_selected_input_columns():

    data = make_data(
        300
    )

    X, y = create_forecasting_windows(
        data,
        input_columns=[
            "pv",
            "load",
        ],
        target_columns=[
            "pv",
        ],
    )

    assert X.shape == (
        109,
        168,
        2,
    )

    assert y.shape == (
        109,
        24,
        1,
    )


def test_too_short_dataset_rejected():

    data = make_data(
        191
    )

    with pytest.raises(ValueError):

        create_forecasting_windows(
            data
        )


# ============================================================
# TIMESTAMPS
# ============================================================

def test_timestamp_shapes():

    data = make_data(
        300
    )

    X_time, y_time = (
        create_window_timestamps(
            data
        )
    )

    assert X_time.shape == (
        109,
        168,
    )

    assert y_time.shape == (
        109,
        24,
    )


def test_first_target_timestamp():

    data = make_data(
        300
    )

    _, y_time = (
        create_window_timestamps(
            data
        )
    )

    expected = np.datetime64(
        data.index[
            168
        ],
        "ns",
    )

    assert y_time[
        0,
        0
    ] == expected


# ============================================================
# DATASET CLASS
# ============================================================

def test_dataset_length():

    data = make_data(
        300
    )

    dataset = ForecastWindowDataset(
        data
    )

    assert len(
        dataset
    ) == 109


def test_dataset_getitem_shapes():

    data = make_data(
        300
    )

    dataset = ForecastWindowDataset(
        data
    )

    X, y = dataset[
        0
    ]

    assert X.shape == (
        168,
        4,
    )

    assert y.shape == (
        24,
        4,
    )


def test_dataset_negative_index():

    data = make_data(
        300
    )

    dataset = ForecastWindowDataset(
        data
    )

    X1, y1 = dataset[
        -1
    ]

    X2, y2 = dataset[
        len(dataset) - 1
    ]

    assert np.array_equal(
        X1,
        X2,
    )

    assert np.array_equal(
        y1,
        y2,
    )


def test_dataset_index_out_of_range():

    data = make_data(
        300
    )

    dataset = ForecastWindowDataset(
        data
    )

    with pytest.raises(IndexError):

        _ = dataset[
            1000
        ]


def test_dataset_feature_counts():

    data = make_data(
        300
    )

    config = ForecastDatasetConfig(
        input_columns=[
            "pv",
            "load",
            "ev",
            "price",
        ],
        target_columns=[
            "pv",
            "load",
        ],
    )

    dataset = ForecastWindowDataset(
        data,
        config,
    )

    assert (
        dataset.number_of_input_features
        == 4
    )

    assert (
        dataset.number_of_target_features
        == 2
    )


def test_dataset_input_shape():

    dataset = ForecastWindowDataset(
        make_data(
            300
        )
    )

    assert dataset.input_shape == (
        168,
        4,
    )


def test_dataset_target_shape():

    dataset = ForecastWindowDataset(
        make_data(
            300
        )
    )

    assert dataset.target_shape == (
        24,
        4,
    )


def test_dataset_timestamps():

    dataset = ForecastWindowDataset(
        make_data(
            300
        )
    )

    X_time, y_time = (
        dataset.get_timestamps(
            0
        )
    )

    assert len(
        X_time
    ) == 168

    assert len(
        y_time
    ) == 24


def test_dataset_summary():

    dataset = ForecastWindowDataset(
        make_data(
            300
        )
    )

    summary = dataset.summary()

    assert summary[
        "number_of_samples"
    ] == 109

    assert summary[
        "input_window"
    ] == 168

    assert summary[
        "forecast_horizon"
    ] == 24


def test_dataset_repr():

    dataset = ForecastWindowDataset(
        make_data(
            300
        )
    )

    text = repr(
        dataset
    )

    assert "ForecastWindowDataset" in text
    assert "168" in text
    assert "24" in text


# ============================================================
# TRAIN / VALIDATION / TEST CONSTRUCTION
# ============================================================

def test_build_three_datasets():

    train = make_data(
        400
    )

    validation = make_data(
        250
    )

    test = make_data(
        250
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = build_forecasting_datasets(
        train,
        validation,
        test,
    )

    assert len(
        train_dataset
    ) > 0

    assert len(
        validation_dataset
    ) > 0

    assert len(
        test_dataset
    ) > 0


def test_split_datasets_are_independent():

    train = make_data(
        400
    )

    validation = make_data(
        250
    )

    test = make_data(
        250
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = build_forecasting_datasets(
        train,
        validation,
        test,
    )

    assert (
        train_dataset.data
        is not validation_dataset.data
    )

    assert (
        validation_dataset.data
        is not test_dataset.data
    )