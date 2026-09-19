"""Tests for the current interval-based forecasting confidence API."""

import numpy as np
import pytest

from forecasting.confidence import (
    ForecastConfidenceConfig,
    calculate_relative_interval_uncertainty,
    calculate_all_feature_uncertainties,
    calculate_adaptive_confidence,
    build_confidence_aware_predictive_state,
    validate_array,
    validate_same_shapes,
)


def test_default_config_valid():
    config = ForecastConfidenceConfig()
    config.validate()
    assert len(config.feature_weights) == 4


def test_invalid_delta_rejected():
    with pytest.raises(ValueError):
        ForecastConfidenceConfig(delta=0.0).validate()


def test_invalid_confidence_bounds_rejected():
    with pytest.raises(ValueError):
        ForecastConfidenceConfig(
            minimum_confidence=0.8,
            maximum_confidence=0.5,
        ).validate()


def test_four_feature_weights_required():
    with pytest.raises(ValueError):
        ForecastConfidenceConfig(
            feature_weights=(0.5, 0.5),
        ).validate()


def test_validate_array_rejects_nan():
    x = np.ones((2, 24, 4))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_array(x, "x")


def test_validate_same_shapes_rejects_mismatch():
    a = np.zeros((2, 24, 4))
    b = np.zeros((2, 12, 4))
    with pytest.raises(ValueError):
        validate_same_shapes(a, b)


def test_zero_width_interval_has_zero_uncertainty():
    forecast = np.ones((2, 24))
    result = calculate_relative_interval_uncertainty(
        forecast,
        forecast,
        forecast,
    )
    assert np.allclose(result, 0.0)


def test_relative_interval_uncertainty_known_value():
    forecast = np.full((2, 24), 10.0)
    lower = np.full((2, 24), 8.0)
    upper = np.full((2, 24), 12.0)

    result = calculate_relative_interval_uncertainty(
        forecast,
        lower,
        upper,
        delta=1e-6,
    )

    expected = 4.0 / (2.0 * (10.0 + 1e-6))
    assert np.allclose(result, expected)


def test_invalid_interval_order_rejected():
    forecast = np.ones((2, 24))
    lower = np.full((2, 24), 2.0)
    upper = np.ones((2, 24))

    with pytest.raises(ValueError):
        calculate_relative_interval_uncertainty(
            forecast,
            lower,
            upper,
        )


def test_all_feature_uncertainty_shape():
    forecast = np.ones((3, 24, 4))
    lower = forecast - 0.1
    upper = forecast + 0.1

    result = calculate_all_feature_uncertainties(
        forecast,
        lower,
        upper,
    )

    assert result.shape == (3, 24, 4)
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)


def test_adaptive_confidence_is_bounded():
    uncertainty = np.array(
        [[0.0, 0.25, 0.5, 1.0, 2.0]],
        dtype=np.float64,
    )

    confidence = calculate_adaptive_confidence(
        uncertainty,
        minimum_confidence=0.1,
        maximum_confidence=0.9,
    )

    assert np.isfinite(confidence).all()
    assert np.all(confidence >= 0.1)
    assert np.all(confidence <= 0.9)


def test_predictive_state_construction():
    forecast = np.ones((2, 24, 4))
    confidence = np.full((2, 24), 0.8)

    state = build_confidence_aware_predictive_state(
        forecast,
        confidence,
    )

    assert state.shape[0] == 2
    assert state.shape[1] == 24
    assert np.isfinite(state).all()
