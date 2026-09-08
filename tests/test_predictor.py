import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from forecasting.predictor import (
    ForecastInputDataset,
    ForecastPredictionResult,
    ForecastPredictor,
    ForecastPredictorConfig,
    create_prediction_loader,
    forecast_to_dataframe,
    inverse_transform_predictions,
    predict_forecasting_model,
    resolve_prediction_device,
    to_float_tensor,
)
# ============================================================
# SMALL MODEL
# ============================================================
class SmallPredictionModel(
    nn.Module
):
    def __init__(
        self,
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                8 * 2,
                3 * 2,
            ),
        )
    def forward(
        self,
        x,
        return_attention=False,
    ):

        forecast = self.network(
            x
        ).reshape(
            x.shape[0],
            3,
            2,
        )

        if return_attention:

            attention = torch.ones(
                x.shape[0],
                8,
                device=x.device,
            )

            attention = (
                attention
                / attention.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            return (
                forecast,
                attention,
            )
        return forecast
# ============================================================
# SYNTHETIC DATASET
# ============================================================

class SmallPredictionDataset:

    def __init__(
        self,
        number_of_samples=10,
    ):

        rng = np.random.default_rng(
            42
        )

        self.X = rng.normal(
            size=(
                number_of_samples,
                8,
                2,
            )
        ).astype(
            np.float32
        )

        self.y = rng.normal(
            size=(
                number_of_samples,
                3,
                2,
            )
        ).astype(
            np.float32
        )

    def __len__(
        self,
    ):

        return len(
            self.X
        )

    def __getitem__(
        self,
        index,
    ):

        return (
            self.X[index],
            self.y[index],
        )
# ============================================================
# SIMPLE TEST SCALER
# ============================================================

class SimpleScaler:
  
    def inverse_transform(
        self,
        dataframe,
    ):

        return (
            dataframe * 10.0
            + 5.0
        )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model():

    torch.manual_seed(
        42
    )

    network = (
        SmallPredictionModel()
    )

    network.eval()

    return network


@pytest.fixture
def dataset():

    return SmallPredictionDataset()


@pytest.fixture
def config():

    return ForecastPredictorConfig(
        batch_size=4,
        device="cpu",
        return_attention=False,
    )
# ============================================================
# CONFIGURATION
# ============================================================

def test_default_batch_size():

    config = ForecastPredictorConfig()

    assert config.batch_size == 128


def test_valid_config():

    config = ForecastPredictorConfig()

    config.validate()


def test_invalid_batch_size():

    config = ForecastPredictorConfig(
        batch_size=0
    )

    with pytest.raises(ValueError):

        config.validate()


def test_invalid_batch_type():

    config = ForecastPredictorConfig(
        batch_size=2.5
    )

    with pytest.raises(TypeError):

        config.validate()
def test_invalid_device():

    config = ForecastPredictorConfig(
        device="tpu"
    )

    with pytest.raises(ValueError):

        config.validate()
# ============================================================
# DEVICE
# ============================================================

def test_cpu_device():

    device = (
        resolve_prediction_device(
            "cpu"
        )
    )

    assert device.type == "cpu"


def test_auto_device():

    device = (
        resolve_prediction_device(
            "auto"
        )
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_invalid_device_resolver():

    with pytest.raises(ValueError):

        resolve_prediction_device(
            "bad_device"
        )
# ============================================================
# TENSOR CONVERSION
# ============================================================

def test_numpy_to_tensor():

    array = np.zeros(
        (
            2,
            8,
            2,
        ),
        dtype=np.float64,
    )

    tensor = to_float_tensor(
        array
    )

    assert isinstance(
        tensor,
        torch.Tensor,
    )

    assert tensor.dtype == torch.float32
def test_tensor_conversion_shape():

    array = np.zeros(
        (
            2,
            8,
            2,
        )
    )

    tensor = to_float_tensor(
        array
    )

    assert tensor.shape == (
        2,
        8,
        2,
    )
def test_nan_input_rejected():

    array = np.zeros(
        (
            1,
            8,
            2,
        )
    )

    array[
        0,
        0,
        0
    ] = np.nan

    with pytest.raises(ValueError):

        to_float_tensor(
            array
        )


# ============================================================
# INPUT DATASET
# ============================================================

def test_array_dataset_length():

    array = np.zeros(
        (
            10,
            8,
            2,
        ),
        dtype=np.float32,
    )

    dataset = ForecastInputDataset(
        array
    )

    assert len(
        dataset
    ) == 10


def test_array_dataset_sample_shape():

    array = np.zeros(
        (
            10,
            8,
            2,
        ),
        dtype=np.float32,
    )

    dataset = ForecastInputDataset(
        array
    )

    sample = dataset[
        0
    ]

    assert sample.shape == (
        8,
        2,
    )


def test_supervised_dataset_uses_only_x(
    dataset,
):
    prediction_dataset = (
        ForecastInputDataset(
            dataset
        )
    )
    sample = prediction_dataset[
        0
    ]
    assert sample.shape == (
        8,
        2,
    )
def test_invalid_array_dimension():

    array = np.zeros(
        (
            8,
            2,
        )
    )

    with pytest.raises(ValueError):

        ForecastInputDataset(
            array
        )


def test_empty_array_rejected():

    array = np.empty(
        (
            0,
            8,
            2,
        )
    )

    with pytest.raises(ValueError):

        ForecastInputDataset(
            array
        )


# ============================================================
# DATA LOADER
# ============================================================

def test_prediction_loader_shape(
    dataset,
):

    loader = create_prediction_loader(
        dataset,
        batch_size=4,
    )

    X = next(
        iter(
            loader
        )
    )

    assert X.shape == (
        4,
        8,
        2,
    )


def test_loader_not_returning_targets(
    dataset,
):

    loader = create_prediction_loader(
        dataset,
        batch_size=4,
    )

    batch = next(
        iter(
            loader
        )
    )

    assert isinstance(
        batch,
        torch.Tensor,
    )


def test_loader_invalid_batch_size(
    dataset,
):

    with pytest.raises(ValueError):

        create_prediction_loader(
            dataset,
            batch_size=0,
        )


# ============================================================
# PREDICTOR CREATION
# ============================================================

def test_predictor_creation(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    assert isinstance(
        predictor,
        ForecastPredictor,
    )

    assert predictor.device.type == "cpu"


def test_predictor_rejects_non_model(
    config,
):

    with pytest.raises(TypeError):

        ForecastPredictor(
            "not_a_model",
            config,
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

def test_batch_prediction_shape(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        5,
        8,
        2,
    ).astype(
        np.float32
    )

    prediction = (
        predictor.predict_batch(
            X
        )
    )

    assert prediction.shape == (
        5,
        3,
        2,
    )


def test_batch_prediction_is_numpy(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        2,
        8,
        2,
    ).astype(
        np.float32
    )

    prediction = (
        predictor.predict_batch(
            X
        )
    )

    assert isinstance(
        prediction,
        np.ndarray,
    )


def test_batch_prediction_finite(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        2,
        8,
        2,
    ).astype(
        np.float32
    )

    prediction = (
        predictor.predict_batch(
            X
        )
    )

    assert np.isfinite(
        prediction
    ).all()


def test_batch_wrong_rank(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        8,
        2,
    ).astype(
        np.float32
    )

    with pytest.raises(ValueError):

        predictor.predict_batch(
            X
        )


# ============================================================
# ATTENTION
# ============================================================

def test_batch_attention_shape(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        4,
        8,
        2,
    ).astype(
        np.float32
    )

    (
        prediction,
        attention,
    ) = predictor.predict_batch(
        X,
        return_attention=True,
    )

    assert prediction.shape == (
        4,
        3,
        2,
    )

    assert attention.shape == (
        4,
        8,
    )


def test_attention_sums_to_one(
    model,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    X = np.random.randn(
        4,
        8,
        2,
    ).astype(
        np.float32
    )

    _, attention = (
        predictor.predict_batch(
            X,
            return_attention=True,
        )
    )

    assert np.allclose(
        attention.sum(
            axis=1
        ),
        1.0,
    )


# ============================================================
# COMPLETE DATASET PREDICTION
# ============================================================

def test_dataset_prediction(
    model,
    dataset,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    result = predictor.predict(
        dataset
    )

    assert isinstance(
        result,
        ForecastPredictionResult,
    )

    assert result.predictions.shape == (
        10,
        3,
        2,
    )


def test_dataset_prediction_names(
    model,
    dataset,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    result = predictor.predict(
        dataset,
        target_names=[
            "PV",
            "Load",
        ],
    )

    assert result.target_names == [
        "PV",
        "Load",
    ]


def test_dataset_attention(
    model,
    dataset,
    config,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    result = predictor.predict(
        dataset,
        return_attention=True,
    )

    assert result.attention.shape == (
        10,
        8,
    )


# ============================================================
# RESULT OBJECT
# ============================================================

def test_result_properties():

    predictions = np.zeros(
        (
            10,
            24,
            4,
        )
    )

    result = ForecastPredictionResult(
        predictions=predictions,
        target_names=[
            "PV",
            "Load",
            "EV",
            "Price",
        ],
    )

    result.validate()

    assert result.number_of_samples == 10

    assert result.forecast_horizon == 24

    assert result.number_of_targets == 4


def test_result_summary():

    predictions = np.zeros(
        (
            5,
            24,
            4,
        )
    )

    result = ForecastPredictionResult(
        predictions=predictions
    )

    summary = result.summary()

    assert summary[
        "number_of_samples"
    ] == 5

    assert summary[
        "forecast_horizon"
    ] == 24

    assert summary[
        "number_of_targets"
    ] == 4


def test_result_invalid_target_names():

    predictions = np.zeros(
        (
            5,
            24,
            4,
        )
    )

    result = ForecastPredictionResult(
        predictions=predictions,
        target_names=[
            "PV",
            "Load",
        ],
    )

    with pytest.raises(ValueError):

        result.validate()


# ============================================================
# INVERSE TRANSFORM
# ============================================================

def test_inverse_transform_shape():

    predictions = np.ones(
        (
            3,
            24,
            2,
        )
    )

    scaler = SimpleScaler()

    result = inverse_transform_predictions(
        predictions,
        scaler,
        [
            "PV",
            "Load",
        ],
    )

    assert result.shape == (
        3,
        24,
        2,
    )


def test_inverse_transform_values():

    predictions = np.zeros(
        (
            1,
            3,
            2,
        )
    )

    scaler = SimpleScaler()

    result = inverse_transform_predictions(
        predictions,
        scaler,
        [
            "PV",
            "Load",
        ],
    )

    assert np.allclose(
        result,
        5.0,
    )


def test_inverse_transform_target_mismatch():

    predictions = np.zeros(
        (
            1,
            3,
            2,
        )
    )

    scaler = SimpleScaler()

    with pytest.raises(ValueError):

        inverse_transform_predictions(
            predictions,
            scaler,
            [
                "PV",
            ],
        )


# ============================================================
# DATAFRAME
# ============================================================

def test_forecast_dataframe():

    predictions = np.zeros(
        (
            2,
            24,
            4,
        )
    )

    dataframe = forecast_to_dataframe(
        predictions,
        [
            "PV",
            "Load",
            "EV",
            "Price",
        ],
        sample_index=0,
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert dataframe.shape == (
        24,
        5,
    )

    assert list(
        dataframe.columns
    ) == [
        "forecast_hour",
        "PV",
        "Load",
        "EV",
        "Price",
    ]


def test_forecast_hours():

    predictions = np.zeros(
        (
            1,
            24,
            4,
        )
    )

    dataframe = forecast_to_dataframe(
        predictions,
        [
            "PV",
            "Load",
            "EV",
            "Price",
        ],
    )

    assert dataframe[
        "forecast_hour"
    ].iloc[
        0
    ] == 1

    assert dataframe[
        "forecast_hour"
    ].iloc[
        -1
    ] == 24


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def test_checkpoint_loading(
    model,
    config,
    tmp_path,
):

    checkpoint_path = (
        tmp_path
        / "model.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epoch":
                3,

            "validation_loss":
                0.1,
        },
        checkpoint_path,
    )

    predictor = ForecastPredictor(
        model,
        config,
    )

    checkpoint = (
        predictor.load_checkpoint(
            checkpoint_path
        )
    )

    assert checkpoint[
        "epoch"
    ] == 3


def test_missing_checkpoint(
    model,
    config,
    tmp_path,
):

    predictor = ForecastPredictor(
        model,
        config,
    )

    with pytest.raises(
        FileNotFoundError
    ):

        predictor.load_checkpoint(
            tmp_path
            / "missing.pt"
        )


# ============================================================
# HIGH-LEVEL HELPER
# ============================================================

def test_prediction_helper(
    model,
    dataset,
    config,
):

    result = predict_forecasting_model(
        model=model,
        data=dataset,
        config=config,
        target_names=[
            "PV",
            "Load",
        ],
    )

    assert isinstance(
        result,
        ForecastPredictionResult,
    )

    assert result.predictions.shape == (
        10,
        3,
        2,
    )
