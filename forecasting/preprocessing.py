
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class PreprocessingConfig:
    """
    Configuration for the forecasting preprocessing pipeline.
    """
    timestamp_column: str = "timestamp"

    hourly_frequency: str = "h"
    hourly_aggregation: str = "mean"

    interpolation_method: str = "linear"

    sigma_threshold: float = 3.0

    training_ratio: float = 0.70
    validation_ratio: float = 0.15
    testing_ratio: float = 0.15

    def validate(self) -> None:
        """Validate preprocessing configuration."""

        if not self.timestamp_column:
            raise ValueError(
                "timestamp_column cannot be empty."
            )

        if self.sigma_threshold <= 0:
            raise ValueError(
                "sigma_threshold must be greater than zero."
            )

        ratios = (
            self.training_ratio
            + self.validation_ratio
            + self.testing_ratio
        )

        if not np.isclose(
            ratios,
            1.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Training, validation, and testing ratios "
                "must sum to 1.0."
            )

        for name, value in [
            ("training_ratio", self.training_ratio),
            ("validation_ratio", self.validation_ratio),
            ("testing_ratio", self.testing_ratio),
        ]:
            if value <= 0 or value >= 1:
                raise ValueError(
                    f"{name} must satisfy 0 < ratio < 1."
                )


# ============================================================
# GENERAL VALIDATION
# ============================================================

def ensure_dataframe(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and copy a pandas DataFrame.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data cannot be empty."
        )

    return data.copy()


def numeric_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return names of numeric columns.
    """

    return list(
        data.select_dtypes(
            include=[np.number]
        ).columns
    )


# ============================================================
# TIMESTAMP PREPARATION
# ============================================================

def prepare_datetime_index(
    data: pd.DataFrame,
    timestamp_column: Optional[str] = "timestamp",
) -> pd.DataFrame:
    """
    Convert timestamp information to a sorted DatetimeIndex.

    The function accepts either:

    1. a timestamp column, or
    2. an already existing DatetimeIndex.

    Duplicate timestamps are retained here; hourly aggregation later
    handles multiple observations within the same hour.
    """

    frame = ensure_dataframe(
        data
    )

    if isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        frame.index = pd.to_datetime(
            frame.index,
            errors="raise",
        )

        return frame.sort_index()

    if timestamp_column is None:
        raise ValueError(
            "A timestamp column is required when "
            "the DataFrame index is not a DatetimeIndex."
        )

    if timestamp_column not in frame.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_column}' was not found."
        )

    timestamps = pd.to_datetime(
        frame[timestamp_column],
        errors="raise",
    )

    frame = frame.drop(
        columns=[timestamp_column]
    )

    frame.index = pd.DatetimeIndex(
        timestamps,
        name=timestamp_column,
    )

    return frame.sort_index()


# ============================================================
# HOURLY RESOLUTION
# ============================================================

def convert_to_hourly(
    data: pd.DataFrame,
    timestamp_column: Optional[str] = "timestamp",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Convert a time-indexed dataset to hourly resolution.

    Parameters
    ----------
    data:
        Input DataFrame.

    timestamp_column:
        Timestamp column when the input does not already use a
        DatetimeIndex.

    aggregation:
        Aggregation used when multiple measurements occur in one hour.

        Supported:
            "mean"
            "sum"
            "median"
            "first"
            "last"

    Notes
    -----
    The manuscript states hourly conversion but does not identify the
    aggregation operator. Therefore this is configurable.
    """

    frame = prepare_datetime_index(
        data=data,
        timestamp_column=timestamp_column,
    )

    supported = {
        "mean",
        "sum",
        "median",
        "first",
        "last",
    }

    if aggregation not in supported:
        raise ValueError(
            f"Unsupported hourly aggregation '{aggregation}'. "
            f"Supported values are {sorted(supported)}."
        )

    # Forecasting variables are numerical.
    columns = numeric_columns(
        frame
    )

    if not columns:
        raise ValueError(
            "No numeric forecasting columns were found."
        )

    numeric_frame = frame[
        columns
    ].copy()

    resampler = numeric_frame.resample(
        "h"
    )

    if aggregation == "mean":
        hourly = resampler.mean()

    elif aggregation == "sum":
        hourly = resampler.sum(
            min_count=1
        )

    elif aggregation == "median":
        hourly = resampler.median()

    elif aggregation == "first":
        hourly = resampler.first()

    else:
        hourly = resampler.last()

    return hourly


# ============================================================
# MISSING-DATA INTERPOLATION
# ============================================================

def interpolate_missing_values(
    data: pd.DataFrame,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Reconstruct missing numerical measurements.

    Manuscript method:
        Linear interpolation.
    """

    frame = ensure_dataframe(
        data
    )

    columns = numeric_columns(
        frame
    )

    if not columns:
        raise ValueError(
            "No numeric columns are available for interpolation."
        )

    result = frame.copy()

    result[columns] = (
        result[columns]
        .interpolate(
            method=method,
            limit_direction="both",
        )
    )

    return result


# ============================================================
# THREE-SIGMA OUTLIER DETECTION
# ============================================================

def three_sigma_mask(
    data: pd.DataFrame,
    sigma_threshold: float = 3.0,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Detect abnormal samples using a three-sigma rule.

    A value x is considered abnormal when:

        |x - mean| > sigma_threshold * standard_deviation

    The default sigma_threshold is 3.0.

    Returns
    -------
    pandas.DataFrame
        Boolean DataFrame with True at abnormal samples.
    """

    frame = ensure_dataframe(
        data
    )

    if sigma_threshold <= 0:
        raise ValueError(
            "sigma_threshold must be greater than zero."
        )

    if columns is None:
        selected_columns = numeric_columns(
            frame
        )
    else:
        selected_columns = list(
            columns
        )

    if not selected_columns:
        raise ValueError(
            "No columns were selected for outlier detection."
        )

    missing_columns = [
        column
        for column in selected_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Columns not found: {missing_columns}"
        )

    mask = pd.DataFrame(
        False,
        index=frame.index,
        columns=frame.columns,
        dtype=bool,
    )

    for column in selected_columns:

        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )

        series = frame[column].astype(
            float
        )

        valid = series.dropna()

        if valid.empty:
            continue

        mean = valid.mean()

        # Population standard deviation is used here as a deterministic
        # reconstruction convention.
        std = valid.std(
            ddof=0
        )

        if (
            not np.isfinite(std)
            or np.isclose(std, 0.0)
        ):
            continue

        lower = (
            mean
            - sigma_threshold * std
        )

        upper = (
            mean
            + sigma_threshold * std
        )

        mask[column] = (
            (series < lower)
            | (series > upper)
        ).fillna(
            False
        )

    return mask


def replace_three_sigma_outliers_with_nan(
    data: pd.DataFrame,
    sigma_threshold: float = 3.0,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect three-sigma outliers and replace those values with NaN.

    Returns
    -------
    cleaned_data:
        DataFrame containing NaN at detected abnormal values.

    outlier_mask:
        Boolean DataFrame identifying detected abnormal values.
    """

    frame = ensure_dataframe(
        data
    )

    mask = three_sigma_mask(
        data=frame,
        sigma_threshold=sigma_threshold,
        columns=columns,
    )

    cleaned = frame.mask(
        mask
    )

    return cleaned, mask


def filter_and_interpolate_outliers(
    data: pd.DataFrame,
    sigma_threshold: float = 3.0,
    interpolation_method: str = "linear",
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete manuscript-consistent abnormal-data treatment.

    Sequence:

        three-sigma detection
                ↓
        abnormal value -> NaN
                ↓
        linear interpolation

    This preserves the regular hourly timeline.
    """

    filtered, mask = (
        replace_three_sigma_outliers_with_nan(
            data=data,
            sigma_threshold=sigma_threshold,
            columns=columns,
        )
    )

    reconstructed = (
        interpolate_missing_values(
            filtered,
            method=interpolation_method,
        )
    )

    return reconstructed, mask


# ============================================================
# MIN-MAX SCALER
# ============================================================

class DataFrameMinMaxScaler:
    """
    Lightweight min-max scaler for pandas DataFrames.

    Transformation:

        x_scaled =
            (x - x_min) / (x_max - x_min)

    Constant columns are mapped to zero.

    The scaler stores training-set min/max values so exactly the same
    transformation can later be applied to validation, test, and
    online observations.
    """

    def __init__(self) -> None:

        self.columns_: Optional[list[str]] = None

        self.minimum_: Optional[pd.Series] = None
        self.maximum_: Optional[pd.Series] = None

        self.range_: Optional[pd.Series] = None

        self.is_fitted_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "DataFrameMinMaxScaler":
        """Fit min/max statistics."""

        frame = ensure_dataframe(
            data
        )

        columns = numeric_columns(
            frame
        )

        if not columns:
            raise ValueError(
                "No numeric columns are available for scaling."
            )

        numeric = frame[
            columns
        ].astype(
            float
        )

        if numeric.isna().any().any():
            raise ValueError(
                "Scaler input contains missing values. "
                "Interpolate missing values before scaling."
            )

        self.columns_ = columns

        self.minimum_ = numeric.min()

        self.maximum_ = numeric.max()

        self.range_ = (
            self.maximum_
            - self.minimum_
        )

        self.is_fitted_ = True

        return self

    def _check_fitted(
        self,
    ) -> None:

        if not self.is_fitted_:
            raise RuntimeError(
                "DataFrameMinMaxScaler has not been fitted."
            )

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply stored min-max transformation.
        """

        self._check_fitted()

        frame = ensure_dataframe(
            data
        )

        assert self.columns_ is not None
        assert self.minimum_ is not None
        assert self.range_ is not None

        missing = [
            column
            for column in self.columns_
            if column not in frame.columns
        ]

        if missing:
            raise KeyError(
                f"Scaler input is missing columns: {missing}"
            )

        result = frame.copy()

        for column in self.columns_:

            denominator = float(
                self.range_[
                    column
                ]
            )

            if np.isclose(
                denominator,
                0.0,
            ):
                result[
                    column
                ] = 0.0

            else:
                result[
                    column
                ] = (
                    result[column]
                    - self.minimum_[column]
                ) / denominator

        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit and transform one dataset."""

        return self.fit(
            data
        ).transform(
            data
        )

    def inverse_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert normalized values back to original units.
        """

        self._check_fitted()

        frame = ensure_dataframe(
            data
        )

        assert self.columns_ is not None
        assert self.minimum_ is not None
        assert self.range_ is not None

        result = frame.copy()

        for column in self.columns_:

            if column not in result.columns:
                raise KeyError(
                    f"Scaler input is missing column '{column}'."
                )

            denominator = float(
                self.range_[
                    column
                ]
            )

            if np.isclose(
                denominator,
                0.0,
            ):
                result[
                    column
                ] = self.minimum_[
                    column
                ]

            else:
                result[
                    column
                ] = (
                    result[column]
                    * denominator
                    + self.minimum_[column]
                )

        return result

    def get_parameters(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return scaler statistics for saving/reproduction.
        """

        self._check_fitted()

        assert self.columns_ is not None
        assert self.minimum_ is not None
        assert self.maximum_ is not None

        return {
            column: {
                "minimum": float(
                    self.minimum_[
                        column
                    ]
                ),
                "maximum": float(
                    self.maximum_[
                        column
                    ]
                ),
            }
            for column in self.columns_
        }


# ============================================================
# CHRONOLOGICAL 70 / 15 / 15 SPLIT
# ============================================================

def chronological_split(
    data: pd.DataFrame,
    training_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    testing_ratio: float = 0.15,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Chronologically divide a time-series dataset.

    Manuscript ratios:

        training   = 70%
        validation = 15%
        testing    = 15%

    No random shuffling is performed.
    """

    frame = ensure_dataframe(
        data
    )

    total_ratio = (
        training_ratio
        + validation_ratio
        + testing_ratio
    )

    if not np.isclose(
        total_ratio,
        1.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Split ratios must sum to 1.0."
        )

    for ratio in [
        training_ratio,
        validation_ratio,
        testing_ratio,
    ]:
        if ratio <= 0 or ratio >= 1:
            raise ValueError(
                "Each split ratio must satisfy 0 < ratio < 1."
            )

    n = len(
        frame
    )

    train_end = int(
        np.floor(
            n * training_ratio
        )
    )

    validation_size = int(
        np.floor(
            n * validation_ratio
        )
    )

    validation_end = (
        train_end
        + validation_size
    )

    train = frame.iloc[
        :train_end
    ].copy()

    validation = frame.iloc[
        train_end:validation_end
    ].copy()

    test = frame.iloc[
        validation_end:
    ].copy()

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise ValueError(
            "Dataset is too small for the requested split ratios."
        )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_forecasting_data(
    data: pd.DataFrame,
    config: Optional[
        PreprocessingConfig
    ] = None,
) -> Dict[str, object]:
    """
    Execute the complete manuscript preprocessing chain.

    Pipeline
    --------
    raw data
        ↓
    datetime index
        ↓
    hourly conversion
        ↓
    linear interpolation of missing measurements
        ↓
    three-sigma abnormal-sample detection
        ↓
    abnormal values -> NaN
        ↓
    linear interpolation
        ↓
    chronological 70/15/15 split
        ↓
    fit min-max scaler on training data
        ↓
    transform train/validation/test

    Returns
    -------
    dict containing:
        hourly_data
        cleaned_data
        outlier_mask
        train_raw
        validation_raw
        test_raw
        train_scaled
        validation_scaled
        test_scaled
        scaler
        config
    """

    if config is None:
        config = PreprocessingConfig()

    config.validate()

    # --------------------------------------------------------
    # HOURLY ALIGNMENT
    # --------------------------------------------------------

    hourly = convert_to_hourly(
        data=data,
        timestamp_column=(
            config.timestamp_column
        ),
        aggregation=(
            config.hourly_aggregation
        ),
    )

    # --------------------------------------------------------
    # MISSING MEASUREMENTS
    # --------------------------------------------------------

    hourly_interpolated = (
        interpolate_missing_values(
            data=hourly,
            method=(
                config.interpolation_method
            ),
        )
    )

    # --------------------------------------------------------
    # THREE-SIGMA FILTER + INTERPOLATION
    # --------------------------------------------------------

    cleaned, outlier_mask = (
        filter_and_interpolate_outliers(
            data=hourly_interpolated,
            sigma_threshold=(
                config.sigma_threshold
            ),
            interpolation_method=(
                config.interpolation_method
            ),
        )
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    (
        train_raw,
        validation_raw,
        test_raw,
    ) = chronological_split(
        data=cleaned,
        training_ratio=(
            config.training_ratio
        ),
        validation_ratio=(
            config.validation_ratio
        ),
        testing_ratio=(
            config.testing_ratio
        ),
    )

    # --------------------------------------------------------
    # MIN-MAX NORMALIZATION
    # --------------------------------------------------------

    scaler = DataFrameMinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_raw
    )

    validation_scaled = scaler.transform(
        validation_raw
    )

    test_scaled = scaler.transform(
        test_raw
    )

    return {
        "hourly_data":
            hourly,

        "cleaned_data":
            cleaned,

        "outlier_mask":
            outlier_mask,

        "train_raw":
            train_raw,

        "validation_raw":
            validation_raw,

        "test_raw":
            test_raw,

        "train_scaled":
            train_scaled,

        "validation_scaled":
            validation_scaled,

        "test_scaled":
            test_scaled,

        "scaler":
            scaler,

        "config":
            config,
    }
