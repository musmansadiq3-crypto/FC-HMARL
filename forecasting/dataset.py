from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ForecastDatasetConfig:
    """
    Configuration for supervised forecasting-window construction.
    """
    input_window: int = 168
    forecast_horizon: int = 24
    stride: int = 1

    input_columns: Optional[Sequence[str]] = None
    target_columns: Optional[Sequence[str]] = None

    def validate(self) -> None:
        """Validate the dataset configuration."""

        if not isinstance(self.input_window, int):
            raise TypeError("input_window must be an integer.")

        if not isinstance(self.forecast_horizon, int):
            raise TypeError("forecast_horizon must be an integer.")

        if not isinstance(self.stride, int):
            raise TypeError("stride must be an integer.")

        if self.input_window <= 0:
            raise ValueError(
                "input_window must be greater than zero."
            )

        if self.forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be greater than zero."
            )

        if self.stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )
# ============================================================
# VALIDATION UTILITIES
# ============================================================

def validate_time_series(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate an input forecasting time series.

    Requirements
    ------------
    - pandas DataFrame
    - non-empty
    - numerical forecasting variables
    - no missing values
    - chronological index when DatetimeIndex is used
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data cannot be empty."
        )

    frame = data.copy()

    if frame.columns.duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate column names."
        )

    numeric = frame.select_dtypes(
        include=[np.number]
    )

    if numeric.shape[1] != frame.shape[1]:
        non_numeric = [
            column
            for column in frame.columns
            if column not in numeric.columns
        ]

        raise TypeError(
            "All forecasting columns must be numeric. "
            f"Non-numeric columns: {non_numeric}"
        )

    if frame.isna().any().any():
        raise ValueError(
            "Forecasting dataset contains missing values. "
            "Run preprocessing before dataset construction."
        )

    values = frame.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Forecasting dataset contains non-finite values."
        )

    if isinstance(frame.index, pd.DatetimeIndex):

        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                "DatetimeIndex must be chronologically sorted."
            )

        if frame.index.has_duplicates:
            raise ValueError(
                "DatetimeIndex cannot contain duplicate timestamps."
            )

    return frame
def resolve_columns(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]],
) -> list[str]:
    """
    Resolve and validate selected forecasting columns.
    If columns is None, all columns are selected.
    """
    if columns is None:
        return list(data.columns)

    selected = list(columns)

    if len(selected) == 0:
        raise ValueError(
            "Column selection cannot be empty."
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "Column selection contains duplicate names."
        )

    missing = [
        column
        for column in selected
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Columns not found in dataset: {missing}"
        )

    return selected
# ============================================================
# SAMPLE COUNT
# ============================================================
def calculate_number_of_samples(
    sequence_length: int,
    input_window: int = 168,
    forecast_horizon: int = 24,
    stride: int = 1,
) -> int:
    if not isinstance(sequence_length, int):
        raise TypeError(
            "sequence_length must be an integer."
        )

    if sequence_length < 0:
        raise ValueError(
            "sequence_length cannot be negative."
        )

    for name, value in [
        ("input_window", input_window),
        ("forecast_horizon", forecast_horizon),
        ("stride", stride),
    ]:
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    minimum_length = (
        input_window
        + forecast_horizon
    )

    if sequence_length < minimum_length:
        return 0

    return (
        (
            sequence_length
            - minimum_length
        )
        // stride
    ) + 1

# ============================================================
# WINDOW CREATION
# ============================================================

def create_forecasting_windows(
    data: pd.DataFrame,
    input_window: int = 168,
    forecast_horizon: int = 24,
    stride: int = 1,
    input_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous time series into forecasting windows.

    Returns
    -------
    X:
        Shape:

        (
            number_of_samples,
            input_window,
            number_of_input_features
        )

    y:
        Shape:

        (
            number_of_samples,
            forecast_horizon,
            number_of_target_features
        )
    """

    frame = validate_time_series(
        data
    )

    if input_window <= 0:
        raise ValueError(
            "input_window must be greater than zero."
        )

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be greater than zero."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than zero."
        )

    inputs = resolve_columns(
        frame,
        input_columns,
    )

    targets = resolve_columns(
        frame,
        target_columns,
    )

    n_samples = calculate_number_of_samples(
        sequence_length=len(frame),
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if n_samples == 0:
        raise ValueError(
            "Dataset is too short to create even one complete "
            "forecasting sample. Required minimum length is "
            f"{input_window + forecast_horizon}, but received "
            f"{len(frame)}."
        )

    input_values = frame[
        inputs
    ].to_numpy(
        dtype=np.float32
    )

    target_values = frame[
        targets
    ].to_numpy(
        dtype=np.float32
    )

    X = np.empty(
        (
            n_samples,
            input_window,
            len(inputs),
        ),
        dtype=np.float32,
    )

    y = np.empty(
        (
            n_samples,
            forecast_horizon,
            len(targets),
        ),
        dtype=np.float32,
    )

    for sample_index in range(n_samples):

        start = (
            sample_index
            * stride
        )

        input_end = (
            start
            + input_window
        )

        target_end = (
            input_end
            + forecast_horizon
        )

        X[
            sample_index
        ] = input_values[
            start:input_end
        ]

        y[
            sample_index
        ] = target_values[
            input_end:target_end
        ]

    return X, y
# ============================================================
# TIMESTAMP WINDOWS
# ============================================================
def create_window_timestamps(
    data: pd.DataFrame,
    input_window: int = 168,
    forecast_horizon: int = 24,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return timestamps corresponding to every X and y window.

    This is useful for evaluation and plotting.

    Requires a DatetimeIndex.
    """

    frame = validate_time_series(
        data
    )

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "create_window_timestamps requires a DatetimeIndex."
        )

    n_samples = calculate_number_of_samples(
        sequence_length=len(frame),
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if n_samples == 0:
        raise ValueError(
            "Dataset is too short for the requested windows."
        )

    X_time = np.empty(
        (
            n_samples,
            input_window,
        ),
        dtype="datetime64[ns]",
    )

    y_time = np.empty(
        (
            n_samples,
            forecast_horizon,
        ),
        dtype="datetime64[ns]",
    )

    timestamps = frame.index.to_numpy(
        dtype="datetime64[ns]"
    )

    for sample_index in range(n_samples):

        start = (
            sample_index
            * stride
        )

        input_end = (
            start
            + input_window
        )

        target_end = (
            input_end
            + forecast_horizon
        )

        X_time[
            sample_index
        ] = timestamps[
            start:input_end
        ]

        y_time[
            sample_index
        ] = timestamps[
            input_end:target_end
        ]

    return X_time, y_time
# ============================================================
# FORECAST DATASET CLASS
# ============================================================

class ForecastWindowDataset:
    """
    In-memory forecasting dataset.

    This class intentionally does not depend on PyTorch.

    The forecasting/model layer can later convert individual NumPy
    samples to torch.Tensor objects. Keeping this stage framework-
    independent makes preprocessing and dataset construction easy to
    test independently.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[
            ForecastDatasetConfig
        ] = None,
    ) -> None:

        if config is None:
            config = ForecastDatasetConfig()

        config.validate()

        self.config = config

        self.data = validate_time_series(
            data
        )

        self.input_columns = resolve_columns(
            self.data,
            config.input_columns,
        )

        self.target_columns = resolve_columns(
            self.data,
            config.target_columns,
        )

        self.X, self.y = create_forecasting_windows(
            data=self.data,
            input_window=config.input_window,
            forecast_horizon=config.forecast_horizon,
            stride=config.stride,
            input_columns=self.input_columns,
            target_columns=self.target_columns,
        )

        if isinstance(
            self.data.index,
            pd.DatetimeIndex,
        ):

            (
                self.X_timestamps,
                self.y_timestamps,
            ) = create_window_timestamps(
                data=self.data,
                input_window=config.input_window,
                forecast_horizon=config.forecast_horizon,
                stride=config.stride,
            )

        else:
            self.X_timestamps = None
            self.y_timestamps = None

    def __len__(
        self,
    ) -> int:
        """Return number of supervised samples."""

        return int(
            self.X.shape[0]
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one (input, target) pair.
        """

        if not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError(
                "Dataset index must be an integer."
            )

        index = int(
            index
        )

        if index < 0:
            index += len(
                self
            )

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                "Dataset index out of range."
            )

        return (
            self.X[index],
            self.y[index],
        )

    @property
    def number_of_input_features(
        self,
    ) -> int:

        return len(
            self.input_columns
        )

    @property
    def number_of_target_features(
        self,
    ) -> int:

        return len(
            self.target_columns
        )

    @property
    def input_shape(
        self,
    ) -> Tuple[int, int]:

        return (
            self.config.input_window,
            self.number_of_input_features,
        )

    @property
    def target_shape(
        self,
    ) -> Tuple[int, int]:

        return (
            self.config.forecast_horizon,
            self.number_of_target_features,
        )

    def get_timestamps(
        self,
        index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return input and target timestamps for one sample.
        """

        if (
            self.X_timestamps is None
            or self.y_timestamps is None
        ):
            raise RuntimeError(
                "Timestamp information is unavailable because "
                "the source data did not use a DatetimeIndex."
            )

        if index < 0:
            index += len(
                self
            )

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                "Dataset index out of range."
            )

        return (
            self.X_timestamps[index],
            self.y_timestamps[index],
        )

    def summary(
        self,
    ) -> dict:
        """Return dataset metadata."""

        return {
            "number_of_samples":
                len(self),

            "input_window":
                self.config.input_window,

            "forecast_horizon":
                self.config.forecast_horizon,

            "stride":
                self.config.stride,

            "input_columns":
                list(self.input_columns),

            "target_columns":
                list(self.target_columns),

            "number_of_input_features":
                self.number_of_input_features,

            "number_of_target_features":
                self.number_of_target_features,

            "input_shape":
                self.input_shape,

            "target_shape":
                self.target_shape,
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "ForecastWindowDataset("
            f"samples={len(self)}, "
            f"input_window={self.config.input_window}, "
            f"forecast_horizon={self.config.forecast_horizon}, "
            f"input_features={self.number_of_input_features}, "
            f"target_features={self.number_of_target_features}"
            ")"
        )


# ============================================================
# TRAIN / VALIDATION / TEST DATASETS
# ============================================================

def build_forecasting_datasets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Optional[
        ForecastDatasetConfig
    ] = None,
) -> Tuple[
    ForecastWindowDataset,
    ForecastWindowDataset,
    ForecastWindowDataset,
]:
    if config is None:
        config = ForecastDatasetConfig()

    config.validate()

    train_dataset = ForecastWindowDataset(
        data=train_data,
        config=config,
    )

    validation_dataset = ForecastWindowDataset(
        data=validation_data,
        config=config,
    )

    test_dataset = ForecastWindowDataset(
        data=test_data,
        config=config,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )
