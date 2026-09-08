import numpy as np
import pandas as pd
import pytest
from forecasting.preprocessing import (
    DataFrameMinMaxScaler,
    PreprocessingConfig,
    chronological_split,
    convert_to_hourly,
    filter_and_interpolate_outliers,
    interpolate_missing_values,
    prepare_datetime_index,
    preprocess_forecasting_data,
    replace_three_sigma_outliers_with_nan,
    three_sigma_mask,
)
# ============================================================
# DATETIME PREPARATION
# ============================================================
def test_prepare_datetime_index():

    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
            ],
            "pv": [
                10.0,
                20.0,
            ],
        }
    )

    result = prepare_datetime_index(
        data
    )

    assert isinstance(
        result.index,
        pd.DatetimeIndex,
    )

    assert "timestamp" not in result.columns


def test_datetime_index_is_sorted():

    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 02:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
            ],
            "load": [
                30.0,
                10.0,
                20.0,
            ],
        }
    )

    result = prepare_datetime_index(
        data
    )

    assert result.index.is_monotonic_increasing


def test_missing_timestamp_column_rejected():

    data = pd.DataFrame(
        {
            "pv": [
                1.0,
                2.0,
            ]
        }
    )
    with pytest.raises(KeyError):

        prepare_datetime_index(
            data,
            timestamp_column="timestamp",
        )
# ============================================================
# HOURLY CONVERSION
# ============================================================
def test_convert_to_hourly_mean():

    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
                "2024-01-01 01:00:00",
                "2024-01-01 01:30:00",
            ],
            "pv": [
                10.0,
                20.0,
                30.0,
                50.0,
            ],
        }
    )
    result = convert_to_hourly(
        data,
        aggregation="mean",
    )
    assert len(
        result
    ) == 2

    assert result.iloc[
        0
    ][
        "pv"
    ] == pytest.approx(
        15.0
    )
    assert result.iloc[
        1
    ][
        "pv"
    ] == pytest.approx(
        40.0
    )
def test_convert_to_hourly_sum():

    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
            ],
            "ev": [
                5.0,
                7.0,
            ],
        }
    )
    result = convert_to_hourly(
        data,
        aggregation="sum",
    )

    assert result.iloc[
        0
    ][
        "ev"
    ] == pytest.approx(
        12.0
    )
def test_invalid_hourly_aggregation_rejected():
    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
            ],
            "pv": [
                10.0,
            ],
        }
    )
    with pytest.raises(ValueError):

        convert_to_hourly(
            data,
            aggregation="unsupported",
        )
# ============================================================
# LINEAR INTERPOLATION
# ============================================================

def test_linear_interpolation_middle_value():

    data = pd.DataFrame(
        {
            "pv": [
                10.0,
                np.nan,
                30.0,
            ]
        }
    )

    result = interpolate_missing_values(
        data
    )

    assert result.iloc[
        1
    ][
        "pv"
    ] == pytest.approx(
        20.0
    )
def test_interpolation_removes_missing_values():

    data = pd.DataFrame(
        {
            "pv": [
                np.nan,
                10.0,
                np.nan,
                30.0,
                np.nan,
            ]
        }
    )
    result = interpolate_missing_values(
        data
    )

    assert not result[
        "pv"
    ].isna().any()
# ============================================================
# THREE-SIGMA FILTER
# ============================================================

def test_three_sigma_detects_extreme_outlier():

    # 100 normal observations plus one very large outlier.
    values = np.concatenate(
        [
            np.ones(100),
            np.array([1000.0]),
        ]
    )

    data = pd.DataFrame(
        {
            "load": values
        }
    )

    mask = three_sigma_mask(
        data,
        sigma_threshold=3.0,
    )

    assert bool(
        mask.iloc[
            -1
        ][
            "load"
        ]
    )
def test_three_sigma_normal_values_not_marked():

    data = pd.DataFrame(
        {
            "load": np.ones(
                100
            )
        }
    )
    mask = three_sigma_mask(
        data
    )
    assert not mask[
        "load"
    ].any()
def test_outlier_replaced_with_nan():

    values = np.concatenate(
        [
            np.ones(100),
            np.array([1000.0]),
        ]
    )
    data = pd.DataFrame(
        {
            "load": values
        }
    )
    cleaned, mask = (
        replace_three_sigma_outliers_with_nan(
            data
        )
    )
    assert bool(
        mask.iloc[
            -1
        ][
            "load"
        ]
    )

    assert pd.isna(
        cleaned.iloc[
            -1
        ][
            "load"
        ]
    )
def test_outlier_filter_and_interpolation():

    # Place the outlier between two valid values so the expected
    # linear interpolation is obvious.
    values = (
        [10.0] * 50
        + [1000.0]
        + [20.0] * 50
    )
    data = pd.DataFrame(
        {
            "load": values
        }
    )
    cleaned, mask = (
        filter_and_interpolate_outliers(
            data,
            sigma_threshold=3.0,
        )
    )

    assert bool(
        mask.iloc[
            50
        ][
            "load"
        ]
    )

    # Linear interpolation between 10 and 20.
    assert cleaned.iloc[
        50
    ][
        "load"
    ] == pytest.approx(
        15.0
    )
def test_invalid_sigma_threshold():
    data = pd.DataFrame(
        {
            "load": [
                1.0,
                2.0,
            ]
        }
    )
    with pytest.raises(ValueError):

        three_sigma_mask(
            data,
            sigma_threshold=0.0,
        )
# ============================================================
# MIN-MAX SCALER
# ============================================================

def test_minmax_scaler():

    data = pd.DataFrame(
        {
            "pv": [
                0.0,
                50.0,
                100.0,
            ]
        }
    )

    scaler = DataFrameMinMaxScaler()

    result = scaler.fit_transform(
        data
    )

    expected = np.array(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    assert np.allclose(
        result[
            "pv"
        ].to_numpy(),
        expected,
    )


def test_minmax_scaler_multiple_columns():

    data = pd.DataFrame(
        {
            "pv": [
                0.0,
                50.0,
                100.0,
            ],
            "load": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )

    scaler = DataFrameMinMaxScaler()

    result = scaler.fit_transform(
        data
    )

    assert result[
        "pv"
    ].min() == pytest.approx(
        0.0
    )

    assert result[
        "pv"
    ].max() == pytest.approx(
        1.0
    )

    assert result[
        "load"
    ].min() == pytest.approx(
        0.0
    )

    assert result[
        "load"
    ].max() == pytest.approx(
        1.0
    )


def test_constant_column_scales_to_zero():

    data = pd.DataFrame(
        {
            "price": [
                0.20,
                0.20,
                0.20,
            ]
        }
    )

    scaler = DataFrameMinMaxScaler()

    result = scaler.fit_transform(
        data
    )

    assert np.allclose(
        result[
            "price"
        ].to_numpy(),
        0.0,
    )


def test_inverse_transform_recovers_data():

    data = pd.DataFrame(
        {
            "pv": [
                0.0,
                50.0,
                100.0,
            ]
        }
    )

    scaler = DataFrameMinMaxScaler()

    scaled = scaler.fit_transform(
        data
    )

    recovered = scaler.inverse_transform(
        scaled
    )

    assert np.allclose(
        recovered[
            "pv"
        ].to_numpy(),
        data[
            "pv"
        ].to_numpy(),
    )


def test_transform_before_fit_rejected():

    scaler = DataFrameMinMaxScaler()

    data = pd.DataFrame(
        {
            "pv": [
                1.0,
                2.0,
            ]
        }
    )

    with pytest.raises(RuntimeError):

        scaler.transform(
            data
        )


def test_scaler_parameters():

    data = pd.DataFrame(
        {
            "pv": [
                10.0,
                20.0,
                30.0,
            ]
        }
    )

    scaler = DataFrameMinMaxScaler()

    scaler.fit(
        data
    )

    parameters = scaler.get_parameters()

    assert parameters[
        "pv"
    ][
        "minimum"
    ] == pytest.approx(
        10.0
    )

    assert parameters[
        "pv"
    ][
        "maximum"
    ] == pytest.approx(
        30.0
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def test_70_15_15_split():

    data = pd.DataFrame(
        {
            "value": np.arange(
                100
            )
        }
    )

    train, validation, test = (
        chronological_split(
            data
        )
    )

    assert len(
        train
    ) == 70

    assert len(
        validation
    ) == 15

    assert len(
        test
    ) == 15


def test_split_preserves_chronological_order():

    data = pd.DataFrame(
        {
            "value": np.arange(
                100
            )
        }
    )

    train, validation, test = (
        chronological_split(
            data
        )
    )

    assert train.iloc[
        -1
    ][
        "value"
    ] == 69

    assert validation.iloc[
        0
    ][
        "value"
    ] == 70

    assert validation.iloc[
        -1
    ][
        "value"
    ] == 84

    assert test.iloc[
        0
    ][
        "value"
    ] == 85


def test_invalid_split_ratios_rejected():

    data = pd.DataFrame(
        {
            "value": np.arange(
                100
            )
        }
    )

    with pytest.raises(ValueError):

        chronological_split(
            data,
            training_ratio=0.80,
            validation_ratio=0.15,
            testing_ratio=0.15,
        )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def test_complete_preprocessing_pipeline():

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=240,
        freq="h",
    )

    data = pd.DataFrame(
        {
            "timestamp": timestamps,

            "pv": np.linspace(
                0.0,
                100.0,
                240,
            ),

            "load": np.linspace(
                200.0,
                400.0,
                240,
            ),

            "ev": np.linspace(
                20.0,
                40.0,
                240,
            ),

            "price": np.linspace(
                0.12,
                0.32,
                240,
            ),
        }
    )

    # Introduce one missing measurement.
    data.loc[
        20,
        "load"
    ] = np.nan

    config = PreprocessingConfig()

    result = preprocess_forecasting_data(
        data=data,
        config=config,
    )

    assert "train_scaled" in result
    assert "validation_scaled" in result
    assert "test_scaled" in result
    assert "scaler" in result

    assert not result[
        "cleaned_data"
    ].isna().any().any()


def test_complete_pipeline_split_sizes():

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=200,
        freq="h",
    )

    data = pd.DataFrame(
        {
            "timestamp": timestamps,

            "pv": np.linspace(
                0.0,
                100.0,
                200,
            ),

            "load": np.linspace(
                200.0,
                400.0,
                200,
            ),
        }
    )

    result = preprocess_forecasting_data(
        data
    )

    assert len(
        result[
            "train_raw"
        ]
    ) == 140

    assert len(
        result[
            "validation_raw"
        ]
    ) == 30

    assert len(
        result[
            "test_raw"
        ]
    ) == 30


def test_training_scaled_range_is_zero_to_one():

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=200,
        freq="h",
    )

    data = pd.DataFrame(
        {
            "timestamp": timestamps,

            "pv": np.linspace(
                0.0,
                100.0,
                200,
            ),

            "load": np.linspace(
                200.0,
                400.0,
                200,
            ),
        }
    )

    result = preprocess_forecasting_data(
        data
    )

    train = result[
        "train_scaled"
    ]

    assert train.min().min() >= -1e-12

    assert train.max().max() <= 1.0 + 1e-12


# ============================================================
# CONFIG VALIDATION
# ============================================================

def test_preprocessing_config_valid():

    config = PreprocessingConfig()

    config.validate()


def test_invalid_config_ratios():

    config = PreprocessingConfig(
        training_ratio=0.80,
        validation_ratio=0.15,
        testing_ratio=0.15,
    )

    with pytest.raises(ValueError):

        config.validate()


def test_invalid_config_sigma():

    config = PreprocessingConfig(
        sigma_threshold=-3.0
    )

    with pytest.raises(ValueError):

        config.validate()
