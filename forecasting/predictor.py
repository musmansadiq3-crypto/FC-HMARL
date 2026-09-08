from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ForecastPredictorConfig:
    """
    Configuration for forecasting inference.
    """
    batch_size: int = 128

    device: str = "auto"

    return_attention: bool = False

    def validate(self) -> None:
        """Validate predictor configuration."""

        if not isinstance(
            self.batch_size,
            int,
        ):
            raise TypeError(
                "batch_size must be an integer."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        supported_devices = {
            "auto",
            "cpu",
            "cuda",
        }

        if self.device.lower() not in (
            supported_devices
        ):
            raise ValueError(
                "device must be one of: "
                "'auto', 'cpu', or 'cuda'."
            )
# ============================================================
# DEVICE
# ============================================================
def resolve_prediction_device(
    requested_device: str = "auto",
) -> torch.device:
    """
    Resolve device used for model inference.
    """
    requested_device = (
        requested_device.lower()
    )

    if requested_device == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    if requested_device == "cpu":

        return torch.device(
            "cpu"
        )

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is "
                "not available."
            )

        return torch.device(
            "cuda"
        )

    raise ValueError(
        "requested_device must be "
        "'auto', 'cpu', or 'cuda'."
    )
# ============================================================
# INPUT CONVERSION
# ============================================================

def to_float_tensor(
    values,
) -> torch.Tensor:
    """
    Convert NumPy/PyTorch input into float32 tensor.
    """

    if isinstance(
        values,
        torch.Tensor,
    ):

        tensor = values.detach().clone().to(
            dtype=torch.float32
        )

    else:

        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            "Prediction input contains "
            "NaN or Inf."
        )

    return tensor
# ============================================================
# PREDICTION-ONLY DATASET
# ============================================================

class ForecastInputDataset(
    Dataset
):
    """
    Dataset used only for model inference.

    This wrapper accepts either:

        a NumPy array with shape
            (samples, input_window, features)

    or an object implementing:

        __len__()
        __getitem__()

    If the wrapped forecasting dataset returns (X, y), only X
    is used during prediction.
    """

    def __init__(
        self,
        data,
    ) -> None:

        if isinstance(
            data,
            np.ndarray,
        ):

            if data.ndim != 3:

                raise ValueError(
                    "NumPy prediction data must "
                    "have shape "
                    "(samples, sequence, features)."
                )

            if data.shape[0] == 0:

                raise ValueError(
                    "Prediction data cannot be empty."
                )

            self.data = data

            self._array_mode = True

            return

        if not hasattr(
            data,
            "__len__",
        ):

            raise TypeError(
                "Prediction data must implement "
                "__len__."
            )

        if not hasattr(
            data,
            "__getitem__",
        ):

            raise TypeError(
                "Prediction data must implement "
                "__getitem__."
            )

        if len(data) <= 0:

            raise ValueError(
                "Prediction dataset cannot be empty."
            )

        self.data = data
        self._array_mode = False

    def __len__(
        self,
    ) -> int:

        return len(
            self.data
        )

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:

        sample = self.data[
            index
        ]

        # ForecastWindowDataset returns:
        #
        #     X, y
        #
        # For inference we need only X.
        if (
            isinstance(
                sample,
                (tuple, list),
            )
            and len(sample) >= 1
        ):

            sample = sample[
                0
            ]

        tensor = to_float_tensor(
            sample
        )

        if tensor.ndim != 2:

            raise ValueError(
                "Each prediction sample must "
                "have shape "
                "(sequence, features)."
            )

        return tensor


# ============================================================
# DATA LOADER
# ============================================================

def create_prediction_loader(
    data,
    batch_size: int = 128,
) -> DataLoader:
    """
    Create deterministic non-shuffled prediction DataLoader.
    """

    if not isinstance(
        batch_size,
        int,
    ):
        raise TypeError(
            "batch_size must be an integer."
        )

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be positive."
        )

    dataset = ForecastInputDataset(
        data
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

@dataclass
class ForecastPredictionResult:
    """
    Container for forecasting-model outputs.

    Attributes
    ----------
    predictions:
        Array with shape:

            (samples,
             forecast_horizon,
             target_features)

    attention:
        Optional array with shape:

            (samples,
             historical_window)

    target_names:
        Optional variable names.
    """

    predictions: np.ndarray

    attention: Optional[
        np.ndarray
    ] = None

    target_names: Optional[
        List[str]
    ] = None

    def validate(
        self,
    ) -> None:

        if not isinstance(
            self.predictions,
            np.ndarray,
        ):
            raise TypeError(
                "predictions must be a "
                "NumPy array."
            )

        if self.predictions.ndim != 3:

            raise ValueError(
                "predictions must have shape "
                "(samples, horizon, features)."
            )

        if not np.isfinite(
            self.predictions
        ).all():

            raise ValueError(
                "predictions contain "
                "NaN or Inf."
            )

        if self.attention is not None:

            if not isinstance(
                self.attention,
                np.ndarray,
            ):
                raise TypeError(
                    "attention must be a "
                    "NumPy array."
                )

            if self.attention.ndim != 2:

                raise ValueError(
                    "attention must have shape "
                    "(samples, input_window)."
                )

            if (
                self.attention.shape[
                    0
                ]
                != self.predictions.shape[
                    0
                ]
            ):

                raise ValueError(
                    "Prediction and attention "
                    "sample counts do not match."
                )

        if self.target_names is not None:

            if (
                len(
                    self.target_names
                )
                != self.predictions.shape[
                    2
                ]
            ):

                raise ValueError(
                    "Number of target names does "
                    "not match forecast feature "
                    "dimension."
                )

    @property
    def number_of_samples(
        self,
    ) -> int:

        return int(
            self.predictions.shape[
                0
            ]
        )

    @property
    def forecast_horizon(
        self,
    ) -> int:

        return int(
            self.predictions.shape[
                1
            ]
        )

    @property
    def number_of_targets(
        self,
    ) -> int:

        return int(
            self.predictions.shape[
                2
            ]
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "number_of_samples":
                self.number_of_samples,

            "forecast_horizon":
                self.forecast_horizon,

            "number_of_targets":
                self.number_of_targets,

            "has_attention":
                self.attention is not None,

            "target_names":
                self.target_names,
        }
# ============================================================
# MAIN PREDICTOR
# ============================================================

class ForecastPredictor:
    """
    Inference interface for the forecasting neural network.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[
            ForecastPredictorConfig
        ] = None,
    ) -> None:

        if not isinstance(
            model,
            nn.Module,
        ):

            raise TypeError(
                "model must be a "
                "torch.nn.Module."
            )

        if config is None:

            config = (
                ForecastPredictorConfig()
            )

        config.validate()

        self.config = config

        self.device = (
            resolve_prediction_device(
                config.device
            )
        )

        self.model = model.to(
            self.device
        )

        self.model.eval()

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> Dict[str, object]:
        """
        Load trained model weights.

        Compatible with checkpoints generated by
        ForecastTrainer.save_checkpoint().
        """

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if (
            "model_state_dict"
            not in checkpoint
        ):

            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        return checkpoint

    # ========================================================
    # SINGLE BATCH
    # ========================================================

    @torch.no_grad()
    def predict_batch(
        self,
        X,
        return_attention: Optional[
            bool
        ] = None,
    ):
        """
        Predict one batch of historical sequences.

        Input
        -----
        X:
            shape:
                (batch,
                 input_window,
                 input_features)

        Output
        ------
        predictions:
            shape:
                (batch,
                 forecast_horizon,
                 target_features)
        """

        if return_attention is None:

            return_attention = (
                self.config
                .return_attention
            )

        X_tensor = to_float_tensor(
            X
        )

        if X_tensor.ndim != 3:

            raise ValueError(
                "Batch input must have shape "
                "(batch, sequence, features)."
            )

        X_tensor = X_tensor.to(
            self.device
        )

        if return_attention:

            try:

                output = self.model(
                    X_tensor,
                    return_attention=True,
                )

            except TypeError as error:

                raise TypeError(
                    "The model does not support "
                    "return_attention=True."
                ) from error

            if not (
                isinstance(
                    output,
                    (tuple, list),
                )
                and len(output) == 2
            ):

                raise ValueError(
                    "Attention-enabled model must "
                    "return "
                    "(prediction, attention)."
                )

            prediction = output[
                0
            ]

            attention = output[
                1
            ]

            self._validate_prediction_tensor(
                prediction
            )

            if attention.ndim != 2:

                raise ValueError(
                    "Attention output must be "
                    "2-dimensional."
                )

            return (
                prediction.detach()
                .cpu()
                .numpy(),
                attention.detach()
                .cpu()
                .numpy(),
            )

        prediction = self.model(
            X_tensor
        )

        self._validate_prediction_tensor(
            prediction
        )

        return (
            prediction.detach()
            .cpu()
            .numpy()
        )

    # ========================================================
    # DATASET PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        data,
        target_names: Optional[
            Sequence[str]
        ] = None,
        return_attention: Optional[
            bool
        ] = None,
    ) -> ForecastPredictionResult:
        """
        Predict all samples in a dataset or array.
        """

        if return_attention is None:

            return_attention = (
                self.config
                .return_attention
            )

        loader = create_prediction_loader(
            data=data,
            batch_size=(
                self.config.batch_size
            ),
        )

        prediction_batches = []

        attention_batches = []

        self.model.eval()

        for X in loader:

            X = X.to(
                self.device
            )

            if return_attention:

                try:

                    output = self.model(
                        X,
                        return_attention=True,
                    )

                except TypeError as error:

                    raise TypeError(
                        "The model does not support "
                        "attention output."
                    ) from error

                if not (
                    isinstance(
                        output,
                        (tuple, list),
                    )
                    and len(output) == 2
                ):

                    raise ValueError(
                        "Attention-enabled prediction "
                        "must return "
                        "(prediction, attention)."
                    )

                prediction = output[
                    0
                ]

                attention = output[
                    1
                ]

                self._validate_prediction_tensor(
                    prediction
                )

                prediction_batches.append(
                    prediction.detach()
                    .cpu()
                    .numpy()
                )

                attention_batches.append(
                    attention.detach()
                    .cpu()
                    .numpy()
                )

            else:

                prediction = self.model(
                    X
                )

                self._validate_prediction_tensor(
                    prediction
                )

                prediction_batches.append(
                    prediction.detach()
                    .cpu()
                    .numpy()
                )

        if len(
            prediction_batches
        ) == 0:

            raise RuntimeError(
                "Prediction produced no "
                "output batches."
            )

        predictions = np.concatenate(
            prediction_batches,
            axis=0,
        )

        attention_array = None

        if return_attention:

            attention_array = np.concatenate(
                attention_batches,
                axis=0,
            )

        names = None

        if target_names is not None:

            names = list(
                target_names
            )

        result = ForecastPredictionResult(
            predictions=predictions,
            attention=attention_array,
            target_names=names,
        )

        result.validate()

        return result

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_prediction_tensor(
        prediction: torch.Tensor,
    ) -> None:

        if not isinstance(
            prediction,
            torch.Tensor,
        ):

            raise TypeError(
                "Model prediction must be "
                "a torch.Tensor."
            )

        if prediction.ndim != 3:

            raise ValueError(
                "Model prediction must have shape "
                "(batch, horizon, features)."
            )

        if not torch.isfinite(
            prediction
        ).all():

            raise FloatingPointError(
                "Model produced NaN or Inf "
                "predictions."
            )


# ============================================================
# INVERSE SCALING
# ============================================================

def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler,
    target_columns: Sequence[str],
) -> np.ndarray:
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if predictions.ndim != 3:

        raise ValueError(
            "predictions must have shape "
            "(samples, horizon, targets)."
        )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            "predictions contain NaN or Inf."
        )

    target_columns = list(
        target_columns
    )

    number_of_targets = (
        predictions.shape[
            2
        ]
    )

    if (
        len(
            target_columns
        )
        != number_of_targets
    ):

        raise ValueError(
            "Number of target_columns must "
            "match prediction feature count."
        )

    if not hasattr(
        scaler,
        "inverse_transform",
    ):

        raise TypeError(
            "scaler must implement "
            "inverse_transform()."
        )

    original_shape = (
        predictions.shape
    )

    flattened = predictions.reshape(
        -1,
        number_of_targets,
    )

    dataframe = pd.DataFrame(
        flattened,
        columns=target_columns,
    )

    inverse = scaler.inverse_transform(
        dataframe
    )

    if isinstance(
        inverse,
        pd.DataFrame,
    ):

        values = inverse[
            target_columns
        ].to_numpy(
            dtype=np.float64
        )

    else:

        values = np.asarray(
            inverse,
            dtype=np.float64,
        )

    if values.shape != (
        flattened.shape
    ):

        raise ValueError(
            "Inverse-transformed values have "
            "an unexpected shape."
        )

    return values.reshape(
        original_shape
    )


# ============================================================
# RESULT TO DATAFRAME
# ============================================================

def forecast_to_dataframe(
    predictions: np.ndarray,
    target_names: Sequence[str],
    sample_index: int = 0,
) -> pd.DataFrame:
    """
    Convert one 24-hour forecast sample into a DataFrame.

    Example output
    --------------
        horizon    PV    Load    EV    Price
        1          ...
        2          ...
        ...
        24         ...
    """

    predictions = np.asarray(
        predictions
    )

    if predictions.ndim != 3:

        raise ValueError(
            "predictions must have shape "
            "(samples, horizon, targets)."
        )

    if not 0 <= sample_index < (
        predictions.shape[
            0
        ]
    ):

        raise IndexError(
            "sample_index is outside "
            "the prediction array."
        )

    target_names = list(
        target_names
    )

    if len(
        target_names
    ) != predictions.shape[
        2
    ]:

        raise ValueError(
            "target_names count does not match "
            "prediction feature count."
        )

    sample = predictions[
        sample_index
    ]

    dataframe = pd.DataFrame(
        sample,
        columns=target_names,
    )

    dataframe.insert(
        0,
        "forecast_hour",
        np.arange(
            1,
            sample.shape[
                0
            ] + 1,
        ),
    )

    return dataframe


# ============================================================
# HIGH-LEVEL HELPER
# ============================================================

def predict_forecasting_model(
    model: nn.Module,
    data,
    config: Optional[
        ForecastPredictorConfig
    ] = None,
    checkpoint_path: Optional[
        str | Path
    ] = None,
    target_names: Optional[
        Sequence[str]
    ] = None,
    return_attention: bool = False,
) -> ForecastPredictionResult:
    """
    High-level helper for trained forecasting inference.
    """

    predictor = ForecastPredictor(
        model=model,
        config=config,
    )

    if checkpoint_path is not None:

        predictor.load_checkpoint(
            checkpoint_path
        )

    return predictor.predict(
        data=data,
        target_names=target_names,
        return_attention=return_attention,
    )
