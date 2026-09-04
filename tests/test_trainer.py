"""
Tests for forecasting/trainer.py.

The tests intentionally use a small neural network and small
synthetic datasets so that the trainer can be verified quickly
without running the full reconstructed transformer.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from forecasting.trainer import (
    ForecastTrainer,
    ForecastTrainerConfig,
    ForecastTrainingHistory,
    TorchForecastDataset,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    create_data_loader,
    resolve_device,
    set_random_seed,
    train_forecasting_model,
)


# ============================================================
# SMALL SYNTHETIC DATASET
# ============================================================

class SmallForecastDataset:
    """
    Synthetic test-only forecasting dataset.
    """

    def __init__(
        self,
        number_of_samples=20,
        input_window=8,
        input_features=2,
        forecast_horizon=3,
        target_features=2,
    ):

        rng = np.random.default_rng(
            42
        )

        self.X = rng.normal(
            size=(
                number_of_samples,
                input_window,
                input_features,
            )
        ).astype(
            np.float32
        )

        self.y = rng.normal(
            size=(
                number_of_samples,
                forecast_horizon,
                target_features,
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
# SMALL MODEL
# ============================================================

class SmallForecastModel(
    nn.Module
):
    """
    Minimal neural network used only to test the trainer.
    """

    def __init__(
        self,
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                8 * 2,
                16,
            ),
            nn.ReLU(),
            nn.Linear(
                16,
                3 * 2,
            ),
        )

    def forward(
        self,
        x,
    ):

        output = self.network(
            x
        )

        return output.reshape(
            x.shape[0],
            3,
            2,
        )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def dataset():

    return SmallForecastDataset()


@pytest.fixture
def model():

    torch.manual_seed(
        42
    )

    return SmallForecastModel()


@pytest.fixture
def trainer_config(
    tmp_path,
):

    return ForecastTrainerConfig(
        optimizer="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=4,
        maximum_epochs=3,
        early_stopping_patience=2,
        minimum_improvement=0.0,
        gradient_clip_norm=1.0,
        scheduler="none",
        loss_function="mse",
        random_seed=42,
        device="cpu",
        checkpoint_directory=str(
            tmp_path
        ),
        checkpoint_filename=(
            "test_forecast.pt"
        ),
    )


# ============================================================
# CONFIGURATION
# ============================================================

def test_default_optimizer():

    config = ForecastTrainerConfig()

    assert config.optimizer == "adam"


def test_default_learning_rate():

    config = ForecastTrainerConfig()

    assert config.learning_rate == pytest.approx(
        1e-4
    )


def test_default_batch_size():

    config = ForecastTrainerConfig()

    assert config.batch_size == 64


def test_valid_configuration():

    config = ForecastTrainerConfig()

    config.validate()


def test_invalid_learning_rate():

    config = ForecastTrainerConfig(
        learning_rate=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_batch_size():

    config = ForecastTrainerConfig(
        batch_size=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_epochs():

    config = ForecastTrainerConfig(
        maximum_epochs=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_optimizer():

    config = ForecastTrainerConfig(
        optimizer="unknown"
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_loss():

    config = ForecastTrainerConfig(
        loss_function="unknown"
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_device():

    config = ForecastTrainerConfig(
        device="tpu"
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_gradient_clip():

    config = ForecastTrainerConfig(
        gradient_clip_norm=0
    )

    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# RANDOM SEED
# ============================================================

def test_random_seed_reproducibility():

    set_random_seed(
        42
    )

    first = torch.randn(
        5
    )

    set_random_seed(
        42
    )

    second = torch.randn(
        5
    )

    assert torch.equal(
        first,
        second,
    )


# ============================================================
# DEVICE
# ============================================================

def test_cpu_device():

    device = resolve_device(
        "cpu"
    )

    assert device.type == "cpu"


def test_auto_device():

    device = resolve_device(
        "auto"
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_invalid_resolve_device():

    with pytest.raises(ValueError):

        resolve_device(
            "invalid"
        )


# ============================================================
# TORCH DATASET
# ============================================================

def test_torch_dataset_length(
    dataset,
):

    torch_dataset = (
        TorchForecastDataset(
            dataset
        )
    )

    assert len(
        torch_dataset
    ) == 20


def test_torch_dataset_shapes(
    dataset,
):

    torch_dataset = (
        TorchForecastDataset(
            dataset
        )
    )

    X, y = torch_dataset[
        0
    ]

    assert X.shape == (
        8,
        2,
    )

    assert y.shape == (
        3,
        2,
    )


def test_torch_dataset_float32(
    dataset,
):

    torch_dataset = (
        TorchForecastDataset(
            dataset
        )
    )

    X, y = torch_dataset[
        0
    ]

    assert X.dtype == torch.float32

    assert y.dtype == torch.float32


# ============================================================
# DATA LOADER
# ============================================================

def test_data_loader_batch_shape(
    dataset,
):

    loader = create_data_loader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    X, y = next(
        iter(
            loader
        )
    )

    assert X.shape == (
        4,
        8,
        2,
    )

    assert y.shape == (
        4,
        3,
        2,
    )


def test_invalid_loader_batch_size(
    dataset,
):

    with pytest.raises(ValueError):

        create_data_loader(
            dataset,
            batch_size=0,
            shuffle=False,
        )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def test_mse_loss():

    loss = build_loss_function(
        "mse"
    )

    assert isinstance(
        loss,
        nn.MSELoss,
    )


def test_mae_loss():

    loss = build_loss_function(
        "mae"
    )

    assert isinstance(
        loss,
        nn.L1Loss,
    )


def test_huber_loss():

    loss = build_loss_function(
        "huber"
    )

    assert isinstance(
        loss,
        nn.SmoothL1Loss,
    )


def test_invalid_loss_builder():

    with pytest.raises(ValueError):

        build_loss_function(
            "bad_loss"
        )


# ============================================================
# OPTIMIZER
# ============================================================

def test_adam_optimizer(
    model,
):

    config = ForecastTrainerConfig(
        optimizer="adam"
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_adamw_optimizer(
    model,
):

    config = ForecastTrainerConfig(
        optimizer="adamw"
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )


# ============================================================
# SCHEDULER
# ============================================================

def test_no_scheduler(
    model,
):

    config = ForecastTrainerConfig(
        scheduler="none"
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert scheduler is None


def test_reduce_on_plateau_scheduler(
    model,
):

    config = ForecastTrainerConfig(
        scheduler="reduce_on_plateau"
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler
        .ReduceLROnPlateau,
    )


# ============================================================
# TRAINER
# ============================================================

def test_trainer_creation(
    model,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    assert isinstance(
        trainer,
        ForecastTrainer,
    )

    assert trainer.device.type == "cpu"


def test_train_single_epoch(
    model,
    dataset,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    loader = create_data_loader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    loss = trainer.train_epoch(
        loader
    )

    assert isinstance(
        loss,
        float,
    )

    assert np.isfinite(
        loss
    )

    assert loss >= 0


def test_validation_epoch(
    model,
    dataset,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    loader = create_data_loader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    loss = trainer.validate_epoch(
        loader
    )

    assert isinstance(
        loss,
        float,
    )

    assert np.isfinite(
        loss
    )


# ============================================================
# COMPLETE TRAINING
# ============================================================

def test_complete_training(
    model,
    dataset,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    history = trainer.fit(
        train_dataset=dataset,
        validation_dataset=dataset,
        verbose=False,
    )

    assert isinstance(
        history,
        ForecastTrainingHistory,
    )

    assert len(
        history.train_loss
    ) >= 1

    assert len(
        history.validation_loss
    ) >= 1

    assert history.best_epoch is not None

    assert np.isfinite(
        history.best_validation_loss
    )


def test_checkpoint_created(
    model,
    dataset,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    trainer.fit(
        train_dataset=dataset,
        validation_dataset=dataset,
        verbose=False,
    )

    assert trainer.checkpoint_path.exists()


# ============================================================
# CHECKPOINT
# ============================================================

def test_checkpoint_load(
    model,
    dataset,
    trainer_config,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    trainer.fit(
        train_dataset=dataset,
        validation_dataset=dataset,
        verbose=False,
    )

    checkpoint = (
        trainer.load_checkpoint()
    )

    assert "model_state_dict" in checkpoint

    assert "optimizer_state_dict" in checkpoint

    assert "epoch" in checkpoint

    assert "validation_loss" in checkpoint


# ============================================================
# HISTORY SAVE
# ============================================================

def test_save_history(
    model,
    dataset,
    trainer_config,
    tmp_path,
):

    trainer = ForecastTrainer(
        model,
        trainer_config,
    )

    trainer.fit(
        train_dataset=dataset,
        validation_dataset=dataset,
        verbose=False,
    )

    history_path = (
        tmp_path
        / "history.json"
    )

    returned_path = (
        trainer.save_history(
            history_path
        )
    )

    assert returned_path.exists()

    with returned_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        contents = json.load(
            file
        )

    assert "train_loss" in contents
    assert "validation_loss" in contents
    assert "best_epoch" in contents


# ============================================================
# TRAINING HELPER
# ============================================================

def test_training_helper(
    model,
    dataset,
    trainer_config,
):

    trained_model, history = (
        train_forecasting_model(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            config=trainer_config,
            verbose=False,
        )
    )

    assert isinstance(
        trained_model,
        nn.Module,
    )

    assert isinstance(
        history,
        ForecastTrainingHistory,
    )


# ============================================================
# HISTORY OBJECT
# ============================================================

def test_history_to_dict():

    history = ForecastTrainingHistory(
        train_loss=[
            1.0,
            0.8,
        ],
        validation_loss=[
            1.1,
            0.9,
        ],
        learning_rate=[
            1e-4,
            1e-4,
        ],
        best_epoch=2,
        best_validation_loss=0.9,
        stopped_early=False,
    )

    result = history.to_dict()

    assert result[
        "best_epoch"
    ] == 2

    assert result[
        "best_validation_loss"
    ] == pytest.approx(
        0.9
    )

    assert len(
        result[
            "train_loss"
        ]
    ) == 2