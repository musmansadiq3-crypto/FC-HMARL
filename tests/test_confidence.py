import numpy as np
import pytest
from forecasting.confidence import (
    ForecastConfidenceConfig,
    ForecastConfidenceEstimator,
    ForecastConfidenceResult,
    build_confidence_aware_predictive_state,
    calculate_confidence_from_errors,
    calculate_ev_market_uncertainty,
    calculate_global_confidence,
    calculate_renewable_load_confidence,
    calculate_renewable_load_uncertainty,
    confidence_from_actual_forecast,
    validate_error_array,
    validate_same_shapes,
)
# ============================================================
# CONFIG
# ============================================================
def test_default_error_scale():
    config = ForecastConfidenceConfig()

    assert config.error_scale == pytest.approx(
        1.0
    )
def test_valid_config():
    config = ForecastConfidenceConfig()
    config.validate()
def test_invalid_error_scale():

    config = ForecastConfidenceConfig(
        error_scale=0.0
    )
    with pytest.raises(ValueError):
        config.validate()
def test_invalid_minimum_confidence():

    config = ForecastConfidenceConfig(
        minimum_confidence=-0.1
    )

    with pytest.raises(ValueError):

        config.validate()
def test_invalid_maximum_confidence():
    config = ForecastConfidenceConfig(
        maximum_confidence=1.1
    )
    with pytest.raises(ValueError):
        config.validate()
def test_minimum_above_maximum():
    config = ForecastConfidenceConfig(
        minimum_confidence=0.8,
        maximum_confidence=0.5,
    )
    with pytest.raises(ValueError):
        config.validate()
# ============================================================
# VALIDATION
# ============================================================

def test_validate_error_array():

    error = np.zeros(
        (
            10,
            24,
        )
    )

    result = validate_error_array(
        error
    )

    assert result.shape == (
        10,
        24,
    )


def test_nan_error_rejected():

    error = np.zeros(
        (
            10,
            24,
        )
    )

    error[
        0,
        0
    ] = np.nan

    with pytest.raises(ValueError):

        validate_error_array(
            error
        )


def test_shape_validation_success():

    a = np.zeros(
        (
            10,
            24,
        )
    )

    b = np.zeros_like(
        a
    )

    validate_same_shapes(
        a,
        b,
    )


def test_shape_validation_failure():

    a = np.zeros(
        (
            10,
            24,
        )
    )

    b = np.zeros(
        (
            10,
            12,
        )
    )

    with pytest.raises(ValueError):

        validate_same_shapes(
            a,
            b,
        )


# ============================================================
# RENEWABLE / LOAD UNCERTAINTY
# ============================================================

def test_zero_renewable_load_uncertainty():

    pv = np.zeros(
        (
            2,
            3,
        )
    )

    load = np.zeros_like(
        pv
    )

    result = (
        calculate_renewable_load_uncertainty(
            pv,
            load,
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


def test_renewable_load_euclidean_error():

    pv = np.full(
        (
            2,
            3,
        ),
        3.0,
    )

    load = np.full(
        (
            2,
            3,
        ),
        4.0,
    )

    result = (
        calculate_renewable_load_uncertainty(
            pv,
            load,
        )
    )

    assert np.allclose(
        result,
        5.0,
    )


def test_negative_errors_still_positive_uncertainty():

    pv = np.full(
        (
            2,
            3,
        ),
        -3.0,
    )

    load = np.full(
        (
            2,
            3,
        ),
        -4.0,
    )

    result = (
        calculate_renewable_load_uncertainty(
            pv,
            load,
        )
    )

    assert np.allclose(
        result,
        5.0,
    )


# ============================================================
# RENEWABLE / LOAD CONFIDENCE
# ============================================================

def test_zero_uncertainty_gives_full_rl_confidence():

    uncertainty = np.zeros(
        (
            5,
            24,
        )
    )

    confidence = (
        calculate_renewable_load_confidence(
            uncertainty
        )
    )

    assert np.allclose(
        confidence,
        1.0,
    )


def test_rl_confidence_matches_exponential():

    uncertainty = np.ones(
        (
            2,
            3,
        )
    )

    confidence = (
        calculate_renewable_load_confidence(
            uncertainty
        )
    )

    assert np.allclose(
        confidence,
        np.exp(
            -1.0
        ),
    )


def test_larger_uncertainty_lower_rl_confidence():

    uncertainty = np.array(
        [
            0.1,
            1.0,
            2.0,
        ]
    )

    confidence = (
        calculate_renewable_load_confidence(
            uncertainty
        )
    )

    assert (
        confidence[
            0
        ]
        >
        confidence[
            1
        ]
        >
        confidence[
            2
        ]
    )


# ============================================================
# EV / MARKET UNCERTAINTY
# ============================================================

def test_ev_market_euclidean_error():

    ev = np.full(
        (
            2,
            3,
        ),
        5.0,
    )

    price = np.full(
        (
            2,
            3,
        ),
        12.0,
    )

    result = (
        calculate_ev_market_uncertainty(
            ev,
            price,
        )
    )

    assert np.allclose(
        result,
        13.0,
    )


def test_zero_ev_market_uncertainty():

    ev = np.zeros(
        (
            2,
            3,
        )
    )

    price = np.zeros_like(
        ev
    )

    result = (
        calculate_ev_market_uncertainty(
            ev,
            price,
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


# ============================================================
# GLOBAL CONFIDENCE
# ============================================================

def test_perfect_forecast_gives_confidence_one():

    shape = (
        4,
        24,
    )

    result = (
        calculate_confidence_from_errors(
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
        )
    )

    assert np.allclose(
        result,
        1.0,
    )


def test_global_confidence_formula():

    omega_rl = np.full(
        (
            2,
            3,
        ),
        0.8,
    )

    epsilon_em = np.full(
        (
            2,
            3,
        ),
        0.5,
    )

    result = (
        calculate_global_confidence(
            omega_rl,
            epsilon_em,
        )
    )

    expected = (
        0.8
        * np.exp(
            -0.5
        )
    )

    assert np.allclose(
        result,
        expected,
    )


def test_global_confidence_between_zero_and_one():

    shape = (
        10,
        24,
    )

    rng = np.random.default_rng(
        42
    )

    result = (
        calculate_confidence_from_errors(
            rng.normal(
                size=shape
            ),
            rng.normal(
                size=shape
            ),
            rng.normal(
                size=shape
            ),
            rng.normal(
                size=shape
            ),
        )
    )

    assert np.all(
        result >= 0
    )

    assert np.all(
        result <= 1
    )


def test_more_error_reduces_global_confidence():

    zero = np.zeros(
        (
            1,
            1,
        )
    )

    small = np.full(
        (
            1,
            1,
        ),
        0.1,
    )

    large = np.full(
        (
            1,
            1,
        ),
        1.0,
    )

    low_error_confidence = (
        calculate_confidence_from_errors(
            small,
            small,
            small,
            small,
        )
    )

    high_error_confidence = (
        calculate_confidence_from_errors(
            large,
            large,
            large,
            large,
        )
    )

    perfect_confidence = (
        calculate_confidence_from_errors(
            zero,
            zero,
            zero,
            zero,
        )
    )

    assert (
        perfect_confidence[
            0,
            0
        ]
        >
        low_error_confidence[
            0,
            0
        ]
        >
        high_error_confidence[
            0,
            0
        ]
    )


# ============================================================
# ESTIMATOR
# ============================================================

def test_confidence_estimator():

    estimator = (
        ForecastConfidenceEstimator()
    )

    error = np.zeros(
        (
            5,
            24,
        )
    )

    result = estimator.calculate(
        pv_error=error,
        load_error=error,
        ev_error=error,
        price_error=error,
    )

    assert isinstance(
        result,
        ForecastConfidenceResult,
    )


def test_estimator_result_shape():

    estimator = (
        ForecastConfidenceEstimator()
    )

    error = np.zeros(
        (
            5,
            24,
        )
    )

    result = estimator.calculate(
        error,
        error,
        error,
        error,
    )

    assert result.shape == (
        5,
        24,
    )


def test_estimator_perfect_confidence():

    estimator = (
        ForecastConfidenceEstimator()
    )

    error = np.zeros(
        (
            5,
            24,
        )
    )

    result = estimator.calculate(
        error,
        error,
        error,
        error,
    )

    assert np.allclose(
        result.global_confidence,
        1.0,
    )


def test_result_summary():

    estimator = (
        ForecastConfidenceEstimator()
    )

    error = np.zeros(
        (
            5,
            24,
        )
    )

    result = estimator.calculate(
        error,
        error,
        error,
        error,
    )

    summary = result.summary()

    assert summary[
        "shape"
    ] == (
        5,
        24,
    )

    assert summary[
        "mean_confidence"
    ] == pytest.approx(
        1.0
    )


# ============================================================
# ACTUAL + FORECAST INTERFACE
# ============================================================

def test_confidence_from_actual_forecast():

    forecast = np.zeros(
        (
            10,
            24,
            4,
        )
    )

    actual = np.zeros_like(
        forecast
    )

    result = (
        confidence_from_actual_forecast(
            actual,
            forecast,
        )
    )

    assert isinstance(
        result,
        ForecastConfidenceResult,
    )

    assert result.global_confidence.shape == (
        10,
        24,
    )


def test_actual_forecast_residual_effect():

    forecast = np.zeros(
        (
            1,
            2,
            4,
        )
    )

    actual = np.zeros_like(
        forecast
    )

    actual[
        0,
        1,
        :
    ] = 1.0

    result = (
        confidence_from_actual_forecast(
            actual,
            forecast,
        )
    )

    assert (
        result.global_confidence[
            0,
            0
        ]
        >
        result.global_confidence[
            0,
            1
        ]
    )


def test_actual_forecast_requires_four_variables():

    actual = np.zeros(
        (
            5,
            24,
            3,
        )
    )

    forecast = np.zeros_like(
        actual
    )

    with pytest.raises(ValueError):

        confidence_from_actual_forecast(
            actual,
            forecast,
        )


def test_actual_forecast_shape_mismatch():

    actual = np.zeros(
        (
            5,
            24,
            4,
        )
    )

    forecast = np.zeros(
        (
            4,
            24,
            4,
        )
    )

    with pytest.raises(ValueError):

        confidence_from_actual_forecast(
            actual,
            forecast,
        )


# ============================================================
# CONFIDENCE-AWARE STATE
# ============================================================

def test_confidence_state_shape():

    forecast = np.ones(
        (
            5,
            24,
            4,
        )
    )

    confidence = np.ones(
        (
            5,
            24,
        )
    )

    result = (
        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )
    )

    assert result.shape == (
        5,
        24,
        4,
    )


def test_full_confidence_preserves_forecast():

    forecast = np.random.default_rng(
        42
    ).random(
        (
            5,
            24,
            4,
        )
    )

    confidence = np.ones(
        (
            5,
            24,
        )
    )

    result = (
        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )
    )

    assert np.allclose(
        result,
        forecast,
    )


def test_zero_confidence_suppresses_forecast():

    forecast = np.ones(
        (
            5,
            24,
            4,
        )
    )

    confidence = np.zeros(
        (
            5,
            24,
        )
    )

    result = (
        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


def test_half_confidence_halves_forecast():

    forecast = np.full(
        (
            2,
            3,
            4,
        ),
        10.0,
    )

    confidence = np.full(
        (
            2,
            3,
        ),
        0.5,
    )

    result = (
        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )
    )

    assert np.allclose(
        result,
        5.0,
    )


def test_state_horizon_mismatch():

    forecast = np.ones(
        (
            5,
            24,
            4,
        )
    )

    confidence = np.ones(
        (
            5,
            12,
        )
    )

    with pytest.raises(ValueError):

        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )


def test_invalid_confidence_above_one():

    forecast = np.ones(
        (
            5,
            24,
            4,
        )
    )

    confidence = np.full(
        (
            5,
            24,
        ),
        1.5,
    )

    with pytest.raises(ValueError):

        build_confidence_aware_predictive_state(
            forecast,
            confidence,
        )
