import numpy as np
import pytest
from forecasting.uncertainty import (
    ForecastUncertaintyConfig,
    ForecastUncertaintyEstimator,
    ForecastUncertaintyResult,
    ResidualQuantileModel,
    calculate_forecast_residuals,
    calculate_interval_coverage,
    calculate_residual_statistics,
    confidence_to_quantiles,
    normalize_uncertainty,
    validate_forecast_arrays,
)

# ============================================================
# SYNTHETIC DATA
# ============================================================

@pytest.fixture
def actual_forecast():

    rng = np.random.default_rng(
        42
    )

    forecast = rng.normal(
        loc=0.5,
        scale=0.1,
        size=(
            100,
            24,
            4,
        ),
    )

    residual = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(
            100,
            24,
            4,
        ),
    )

    actual = (
        forecast
        + residual
    )

    return actual, forecast

# ============================================================
# CONFIGURATION
# ============================================================

def test_default_confidence_level():

    config = ForecastUncertaintyConfig()

    assert config.confidence_level == pytest.approx(
        0.95
    )


def test_valid_config():

    config = ForecastUncertaintyConfig()

    config.validate()

def test_invalid_zero_confidence():

    config = ForecastUncertaintyConfig(
        confidence_level=0.0
    )

    with pytest.raises(ValueError):

        config.validate()


def test_invalid_one_confidence():

    config = ForecastUncertaintyConfig(
        confidence_level=1.0
    )

    with pytest.raises(ValueError):

        config.validate()


def test_invalid_minimum_samples():

    config = ForecastUncertaintyConfig(
        minimum_residual_samples=0
    )

    with pytest.raises(ValueError):

        config.validate()


# ============================================================
# ARRAY VALIDATION
# ============================================================

def test_validate_arrays(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    a, f = validate_forecast_arrays(
        actual,
        forecast,
    )

    assert a.shape == (
        100,
        24,
        4,
    )

    assert f.shape == (
        100,
        24,
        4,
    )


def test_shape_mismatch():

    actual = np.zeros(
        (
            10,
            24,
            4,
        )
    )

    forecast = np.zeros(
        (
            10,
            24,
            3,
        )
    )

    with pytest.raises(ValueError):

        validate_forecast_arrays(
            actual,
            forecast,
        )


def test_wrong_dimension():

    actual = np.zeros(
        (
            10,
            24,
        )
    )

    forecast = np.zeros(
        (
            10,
            24,
        )
    )

    with pytest.raises(ValueError):

        validate_forecast_arrays(
            actual,
            forecast,
        )


def test_nan_actual_rejected():

    actual = np.zeros(
        (
            10,
            24,
            4,
        )
    )

    forecast = np.zeros_like(
        actual
    )

    actual[
        0,
        0,
        0
    ] = np.nan

    with pytest.raises(ValueError):

        validate_forecast_arrays(
            actual,
            forecast,
        )

# ============================================================
# RESIDUALS
# ============================================================

def test_residual_definition():

    forecast = np.ones(
        (
            2,
            3,
            1,
        )
    )

    actual = forecast + 2.0

    residuals = (
        calculate_forecast_residuals(
            actual,
            forecast,
        )
    )

    assert np.allclose(
        residuals,
        2.0,
    )


def test_residual_shape(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    residuals = (
        calculate_forecast_residuals(
            actual,
            forecast,
        )
    )

    assert residuals.shape == (
        100,
        24,
        4,
    )

# ============================================================
# RESIDUAL STATISTICS
# ============================================================

def test_residual_statistics_shape(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    residuals = (
        calculate_forecast_residuals(
            actual,
            forecast,
        )
    )

    statistics = (
        calculate_residual_statistics(
            residuals
        )
    )

    for value in (
        statistics.values()
    ):

        assert value.shape == (
            24,
            4,
        )

def test_residual_statistics_keys(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    residuals = (
        calculate_forecast_residuals(
            actual,
            forecast,
        )
    )

    statistics = (
        calculate_residual_statistics(
            residuals
        )
    )

    assert set(
        statistics.keys()
    ) == {
        "mean",
        "std",
        "mae",
        "rmse",
    }

# ============================================================
# CONFIDENCE QUANTILES
# ============================================================

def test_95_percent_quantiles():

    lower, upper = (
        confidence_to_quantiles(
            0.95
        )
    )

    assert lower == pytest.approx(
        0.025
    )

    assert upper == pytest.approx(
        0.975
    )


def test_90_percent_quantiles():

    lower, upper = (
        confidence_to_quantiles(
            0.90
        )
    )

    assert lower == pytest.approx(
        0.05
    )

    assert upper == pytest.approx(
        0.95
    )


# ============================================================
# ESTIMATOR FIT
# ============================================================

def test_estimator_fit(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    model = estimator.fit(
        actual,
        forecast,
        target_names=[
            "PV",
            "Load",
            "EV",
            "Price",
        ],
    )

    assert isinstance(
        model,
        ResidualQuantileModel,
    )


def test_fitted_quantile_shape(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    model = estimator.fit(
        actual,
        forecast,
    )

    assert model.lower_quantile.shape == (
        24,
        4,
    )

    assert model.upper_quantile.shape == (
        24,
        4,
    )


def test_lower_not_above_upper(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    model = estimator.fit(
        actual,
        forecast,
    )

    assert np.all(
        model.lower_quantile
        <= model.upper_quantile
    )


def test_insufficient_samples():

    actual = np.zeros(
        (
            5,
            24,
            4,
        )
    )

    forecast = np.zeros_like(
        actual
    )

    estimator = (
        ForecastUncertaintyEstimator(
            ForecastUncertaintyConfig(
                minimum_residual_samples=10
            )
        )
    )

    with pytest.raises(ValueError):

        estimator.fit(
            actual,
            forecast,
        )


def test_target_name_mismatch(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    with pytest.raises(ValueError):

        estimator.fit(
            actual,
            forecast,
            target_names=[
                "PV",
                "Load",
            ],
        )


# ============================================================
# TRANSFORM
# ============================================================

def test_transform_before_fit():

    estimator = (
        ForecastUncertaintyEstimator()
    )

    forecasts = np.zeros(
        (
            10,
            24,
            4,
        )
    )

    with pytest.raises(RuntimeError):

        estimator.transform(
            forecasts
        )


def test_transform_shape(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    estimator.fit(
        actual,
        forecast,
    )

    result = estimator.transform(
        forecast[:10]
    )

    assert isinstance(
        result,
        ForecastUncertaintyResult,
    )

    assert result.point_forecast.shape == (
        10,
        24,
        4,
    )

    assert result.lower_bound.shape == (
        10,
        24,
        4,
    )

    assert result.upper_bound.shape == (
        10,
        24,
        4,
    )


def test_interval_width_nonnegative(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    estimator.fit(
        actual,
        forecast,
    )

    result = estimator.transform(
        forecast[:10]
    )

    assert np.all(
        result.interval_width >= 0
    )


def test_uncertainty_std_nonnegative(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    estimator.fit(
        actual,
        forecast,
    )

    result = estimator.transform(
        forecast[:10]
    )

    assert np.all(
        result.uncertainty_std >= 0
    )


def test_wrong_transform_horizon(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    estimator.fit(
        actual,
        forecast,
    )

    wrong = np.zeros(
        (
            5,
            12,
            4,
        )
    )

    with pytest.raises(ValueError):

        estimator.transform(
            wrong
        )


def test_wrong_transform_targets(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    estimator.fit(
        actual,
        forecast,
    )

    wrong = np.zeros(
        (
            5,
            24,
            3,
        )
    )

    with pytest.raises(ValueError):

        estimator.transform(
            wrong
        )


# ============================================================
# FIT TRANSFORM
# ============================================================

def test_fit_transform(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    result = estimator.fit_transform(
        actual,
        forecast,
    )

    assert isinstance(
        result,
        ForecastUncertaintyResult,
    )


# ============================================================
# RESULT SUMMARY
# ============================================================

def test_uncertainty_summary(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    result = estimator.fit_transform(
        actual,
        forecast,
        target_names=[
            "PV",
            "Load",
            "EV",
            "Price",
        ],
    )

    summary = result.summary()

    assert summary[
        "number_of_samples"
    ] == 100

    assert summary[
        "forecast_horizon"
    ] == 24

    assert summary[
        "number_of_targets"
    ] == 4

    assert summary[
        "confidence_level"
    ] == pytest.approx(
        0.95
    )


# ============================================================
# INTERVAL COVERAGE
# ============================================================

def test_perfect_interval_coverage():

    actual = np.ones(
        (
            5,
            3,
            2,
        )
    )

    lower = np.zeros_like(
        actual
    )

    upper = np.full_like(
        actual,
        2.0,
    )

    coverage = (
        calculate_interval_coverage(
            actual,
            lower,
            upper,
        )
    )

    assert coverage == pytest.approx(
        1.0
    )


def test_zero_interval_coverage():

    actual = np.full(
        (
            5,
            3,
            2,
        ),
        10.0,
    )

    lower = np.zeros_like(
        actual
    )

    upper = np.ones_like(
        actual
    )

    coverage = (
        calculate_interval_coverage(
            actual,
            lower,
            upper,
        )
    )

    assert coverage == pytest.approx(
        0.0
    )


def test_coverage_bounds(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    estimator = (
        ForecastUncertaintyEstimator()
    )

    result = estimator.fit_transform(
        actual,
        forecast,
    )

    coverage = (
        calculate_interval_coverage(
            actual,
            result.lower_bound,
            result.upper_bound,
        )
    )

    assert 0.0 <= coverage <= 1.0


# ============================================================
# ALTERNATIVE QUANTILE STRUCTURES
# ============================================================

def test_target_specific_only(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    config = ForecastUncertaintyConfig(
        use_horizon_specific_quantiles=False,
        use_target_specific_quantiles=True,
    )

    estimator = (
        ForecastUncertaintyEstimator(
            config
        )
    )

    model = estimator.fit(
        actual,
        forecast,
    )

    assert model.lower_quantile.shape == (
        24,
        4,
    )


def test_horizon_specific_only(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    config = ForecastUncertaintyConfig(
        use_horizon_specific_quantiles=True,
        use_target_specific_quantiles=False,
    )

    estimator = (
        ForecastUncertaintyEstimator(
            config
        )
    )

    model = estimator.fit(
        actual,
        forecast,
    )

    assert model.lower_quantile.shape == (
        24,
        4,
    )


def test_global_quantiles(
    actual_forecast,
):

    actual, forecast = (
        actual_forecast
    )

    config = ForecastUncertaintyConfig(
        use_horizon_specific_quantiles=False,
        use_target_specific_quantiles=False,
    )

    estimator = (
        ForecastUncertaintyEstimator(
            config
        )
    )

    model = estimator.fit(
        actual,
        forecast,
    )

    assert model.lower_quantile.shape == (
        24,
        4,
    )

    assert np.allclose(
        model.lower_quantile,
        model.lower_quantile[
            0,
            0
        ],
    )


# ============================================================
# NORMALIZED UNCERTAINTY
# ============================================================

def test_normalize_uncertainty_range():

    uncertainty = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    normalized = normalize_uncertainty(
        uncertainty
    )

    assert np.min(
        normalized
    ) == pytest.approx(
        0.0
    )

    assert np.max(
        normalized
    ) == pytest.approx(
        1.0
    )


def test_constant_uncertainty_normalization():

    uncertainty = np.ones(
        (
            5,
            24,
            4,
        )
    )

    normalized = normalize_uncertainty(
        uncertainty
    )

    assert np.allclose(
        normalized,
        0.0,
    )


def test_negative_uncertainty_rejected():

    uncertainty = np.array(
        [
            1.0,
            -1.0,
        ]
    )

    with pytest.raises(ValueError):

        normalize_uncertainty(
            uncertainty
        )
