import pytest
import torch
from forecasting.model import (
    ForecastModelConfig,
    LearnablePositionalEncoding,
    MultiHorizonTransformerForecaster,
    SinusoidalPositionalEncoding,
    build_forecasting_model,
)
@pytest.fixture
def config():

    return ForecastModelConfig(
        input_window=168,
        forecast_horizon=24,
        input_features=4,
        target_features=4,
        hidden_dimension=32,
        number_of_attention_heads=4,
        number_of_encoder_layers=1,
        feedforward_dimension=64,
        dropout=0.0,
        activation="gelu",
        use_learnable_positional_encoding=True,
    )

@pytest.fixture
def model(
    config,
):

    torch.manual_seed(
        42
    )

    network = (
        MultiHorizonTransformerForecaster(
            config
        )
    )

    network.eval()

    return network
# ============================================================
# DEFAULT MANUSCRIPT DIMENSIONS
# ============================================================

def test_default_input_window():

    config = ForecastModelConfig()

    assert config.input_window == 168


def test_default_forecast_horizon():

    config = ForecastModelConfig()

    assert config.forecast_horizon == 24


def test_default_input_features():

    config = ForecastModelConfig()

    assert config.input_features == 4


def test_default_target_features():

    config = ForecastModelConfig()

    assert config.target_features == 4
# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def test_valid_config():

    config = ForecastModelConfig()

    config.validate()


def test_invalid_input_window():

    config = ForecastModelConfig(
        input_window=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_forecast_horizon():

    config = ForecastModelConfig(
        forecast_horizon=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_input_features():

    config = ForecastModelConfig(
        input_features=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_target_features():

    config = ForecastModelConfig(
        target_features=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_hidden_dimension_must_match_heads():

    config = ForecastModelConfig(
        hidden_dimension=30,
        number_of_attention_heads=4,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_dropout_negative():

    config = ForecastModelConfig(
        dropout=-0.1
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_dropout_one():

    config = ForecastModelConfig(
        dropout=1.0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_activation():

    config = ForecastModelConfig(
        activation="sigmoid"
    )

    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# LEARNABLE POSITIONAL ENCODING
# ============================================================

def test_learnable_position_shape():

    encoder = LearnablePositionalEncoding(
        maximum_sequence_length=168,
        hidden_dimension=32,
    )

    x = torch.zeros(
        2,
        168,
        32,
    )

    output = encoder(
        x
    )

    assert output.shape == (
        2,
        168,
        32,
    )


def test_learnable_position_is_trainable():

    encoder = LearnablePositionalEncoding(
        maximum_sequence_length=168,
        hidden_dimension=32,
    )

    assert (
        encoder.position_embedding
        .requires_grad
    )


def test_position_encoding_rejects_long_sequence():

    encoder = LearnablePositionalEncoding(
        maximum_sequence_length=168,
        hidden_dimension=32,
    )

    x = torch.zeros(
        1,
        169,
        32,
    )

    with pytest.raises(ValueError):

        encoder(
            x
        )


# ============================================================
# SINUSOIDAL POSITIONAL ENCODING
# ============================================================

def test_sinusoidal_position_shape():

    encoder = SinusoidalPositionalEncoding(
        maximum_sequence_length=168,
        hidden_dimension=32,
    )

    x = torch.zeros(
        2,
        168,
        32,
    )

    output = encoder(
        x
    )

    assert output.shape == (
        2,
        168,
        32,
    )


def test_sinusoidal_encoding_has_no_trainable_position_parameter():

    encoder = SinusoidalPositionalEncoding(
        maximum_sequence_length=168,
        hidden_dimension=32,
    )

    names = [
        name
        for name, _ in encoder.named_parameters()
    ]

    assert "position_encoding" not in names


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def test_model_constructs(
    model,
):

    assert isinstance(
        model,
        MultiHorizonTransformerForecaster,
    )


def test_model_has_parameters(
    model,
):

    assert (
        model.number_of_parameters()
        > 0
    )


# ============================================================
# FORWARD PASS
# ============================================================

def test_single_batch_output_shape(
    model,
):

    x = torch.randn(
        1,
        168,
        4,
    )

    with torch.no_grad():

        output = model(
            x
        )

    assert output.shape == (
        1,
        24,
        4,
    )


def test_multi_batch_output_shape(
    model,
):

    x = torch.randn(
        8,
        168,
        4,
    )

    with torch.no_grad():

        output = model(
            x
        )

    assert output.shape == (
        8,
        24,
        4,
    )


def test_output_is_finite(
    model,
):

    x = torch.randn(
        4,
        168,
        4,
    )

    with torch.no_grad():

        output = model(
            x
        )

    assert torch.isfinite(
        output
    ).all()


# ============================================================
# ENCODER
# ============================================================

def test_encoder_shape(
    model,
):

    x = torch.randn(
        3,
        168,
        4,
    )

    with torch.no_grad():

        encoded = model.encode(
            x
        )

    assert encoded.shape == (
        3,
        168,
        32,
    )


# ============================================================
# TEMPORAL ATTENTION
# ============================================================

def test_attention_shape(
    model,
):

    x = torch.randn(
        2,
        168,
        4,
    )

    with torch.no_grad():

        (
            forecast,
            attention,
        ) = model(
            x,
            return_attention=True,
        )

    assert forecast.shape == (
        2,
        24,
        4,
    )

    assert attention.shape == (
        2,
        168,
    )


def test_attention_sums_to_one(
    model,
):

    x = torch.randn(
        2,
        168,
        4,
    )

    with torch.no_grad():

        _, attention = model(
            x,
            return_attention=True,
        )

    sums = attention.sum(
        dim=1
    )

    assert torch.allclose(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-6,
    )


def test_attention_nonnegative(
    model,
):

    x = torch.randn(
        2,
        168,
        4,
    )

    with torch.no_grad():

        _, attention = model(
            x,
            return_attention=True,
        )

    assert torch.all(
        attention >= 0.0
    )


# ============================================================
# DIFFERENT TARGET FEATURE COUNT
# ============================================================

def test_single_target_variable():

    config = ForecastModelConfig(
        input_window=168,
        forecast_horizon=24,
        input_features=4,
        target_features=1,
        hidden_dimension=32,
        number_of_attention_heads=4,
        number_of_encoder_layers=1,
        feedforward_dimension=64,
        dropout=0.0,
    )

    model = (
        MultiHorizonTransformerForecaster(
            config
        )
    )

    model.eval()

    x = torch.randn(
        2,
        168,
        4,
    )

    with torch.no_grad():

        output = model(
            x
        )

    assert output.shape == (
        2,
        24,
        1,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_wrong_sequence_length_rejected(
    model,
):

    x = torch.randn(
        2,
        100,
        4,
    )

    with pytest.raises(ValueError):

        model(
            x
        )


def test_wrong_feature_count_rejected(
    model,
):

    x = torch.randn(
        2,
        168,
        3,
    )

    with pytest.raises(ValueError):

        model(
            x
        )


def test_wrong_tensor_dimension_rejected(
    model,
):

    x = torch.randn(
        168,
        4,
    )

    with pytest.raises(ValueError):

        model(
            x
        )


def test_nan_input_rejected(
    model,
):

    x = torch.randn(
        1,
        168,
        4,
    )

    x[
        0,
        10,
        2
    ] = float(
        "nan"
    )

    with pytest.raises(ValueError):

        model(
            x
        )


def test_inf_input_rejected(
    model,
):

    x = torch.randn(
        1,
        168,
        4,
    )

    x[
        0,
        10,
        2
    ] = float(
        "inf"
    )

    with pytest.raises(ValueError):

        model(
            x
        )


# ============================================================
# GRADIENT TEST
# ============================================================

def test_model_supports_backpropagation(
    config,
):

    model = (
        MultiHorizonTransformerForecaster(
            config
        )
    )

    model.train()

    x = torch.randn(
        2,
        168,
        4,
    )

    target = torch.randn(
        2,
        24,
        4,
    )

    output = model(
        x
    )

    loss = torch.mean(
        (
            output
            - target
        ) ** 2
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist


# ============================================================
# FACTORY
# ============================================================

def test_model_factory():

    model = build_forecasting_model(
        input_features=4,
        target_features=4,
        input_window=168,
        forecast_horizon=24,
        hidden_dimension=32,
        number_of_attention_heads=4,
        number_of_encoder_layers=1,
        feedforward_dimension=64,
        dropout=0.0,
    )

    assert isinstance(
        model,
        MultiHorizonTransformerForecaster,
    )


def test_factory_output_shape():

    model = build_forecasting_model(
        input_features=4,
        target_features=4,
        hidden_dimension=32,
        number_of_attention_heads=4,
        number_of_encoder_layers=1,
        feedforward_dimension=64,
        dropout=0.0,
    )

    model.eval()

    x = torch.randn(
        1,
        168,
        4,
    )

    with torch.no_grad():

        output = model(
            x
        )

    assert output.shape == (
        1,
        24,
        4,
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

def test_model_summary(
    model,
):

    summary = model.model_summary()

    assert summary[
        "model_class"
    ] == (
        "MultiHorizonTransformerForecaster"
    )

    assert summary[
        "expected_input_shape"
    ] == (
        None,
        168,
        4,
    )

    assert summary[
        "expected_output_shape"
    ] == (
        None,
        24,
        4,
    )

    assert summary[
        "trainable_parameters"
    ] > 0
# ============================================================
# REPRESENTATION
# ============================================================

def test_model_repr(
    model,
):

    text = repr(
        model
    )

    assert (
        "MultiHorizonTransformerForecaster"
        in text
    )

    assert "input_window=168" in text

    assert "horizon=24" in text

    assert "heads=4" in text
