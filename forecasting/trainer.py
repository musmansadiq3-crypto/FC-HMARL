"""
Training utilities for the FC-HMARL forecasting module.

This module trains the reconstructed multi-horizon forecasting model.

Important manuscript / reconstruction distinction
--------------------------------------------------
The manuscript defines:

    historical input window = 168 h
    forecast horizon        = 24 h

and describes uncertainty-aware multi-horizon forecasting.

However, it does not separately report the complete neural-network
training procedure for the forecasting model.

Therefore the following are explicit reconstruction choices:

    optimizer
    forecasting batch size
    number of epochs
    early stopping
    gradient clipping
    learning-rate scheduling
    checkpointing strategy

These choices are implemented transparently and remain configurable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ForecastTrainerConfig:
    """
    Configuration for forecasting-model training.

    Unless otherwise documented, these parameters are
    reconstruction choices rather than recovered original-code
    hyperparameters.
    """

    optimizer: str = "adam"

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    batch_size: int = 64
    maximum_epochs: int = 100

    early_stopping_patience: int = 15
    minimum_improvement: float = 1e-6

    gradient_clip_norm: Optional[float] = 1.0

    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1e-6

    loss_function: str = "mse"

    random_seed: int = 42

    device: str = "auto"

    checkpoint_directory: str = "outputs/checkpoints"

    checkpoint_filename: str = (
        "best_forecasting_model.pt"
    )

    def validate(self) -> None:
        """Validate training configuration."""

        supported_optimizers = {
            "adam",
            "adamw",
        }

        if self.optimizer.lower() not in (
            supported_optimizers
        ):
            raise ValueError(
                f"Unsupported optimizer "
                f"'{self.optimizer}'. "
                f"Supported optimizers are "
                f"{sorted(supported_optimizers)}."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.maximum_epochs <= 0:
            raise ValueError(
                "maximum_epochs must be positive."
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience cannot be negative."
            )

        if self.minimum_improvement < 0:
            raise ValueError(
                "minimum_improvement cannot be negative."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive "
                "or None."
            )

        supported_schedulers = {
            "none",
            "reduce_on_plateau",
        }

        if self.scheduler.lower() not in (
            supported_schedulers
        ):
            raise ValueError(
                f"Unsupported scheduler "
                f"'{self.scheduler}'."
            )

        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError(
                "scheduler_factor must satisfy "
                "0 < scheduler_factor < 1."
            )

        if self.scheduler_patience < 0:
            raise ValueError(
                "scheduler_patience cannot be negative."
            )

        if self.minimum_learning_rate < 0:
            raise ValueError(
                "minimum_learning_rate cannot be negative."
            )

        supported_losses = {
            "mse",
            "mae",
            "huber",
        }

        if self.loss_function.lower() not in (
            supported_losses
        ):
            raise ValueError(
                f"Unsupported loss function "
                f"'{self.loss_function}'."
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

        if not isinstance(
            self.random_seed,
            int,
        ):
            raise TypeError(
                "random_seed must be an integer."
            )

        checkpoint_path = Path(
            self.checkpoint_filename
        )

        if checkpoint_path.name != (
            self.checkpoint_filename
        ):
            raise ValueError(
                "checkpoint_filename must contain "
                "only a filename, not a path."
            )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible training.
    """

    if not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "seed must be an integer."
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    requested_device: str = "auto",
) -> torch.device:
    """
    Determine the PyTorch execution device.
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
# DATASET ADAPTER
# ============================================================

class TorchForecastDataset(
    Dataset
):
    """
    Adapt a NumPy-style forecasting dataset to PyTorch.

    The wrapped dataset must return:

        X:
            (input_window, input_features)

        y:
            (forecast_horizon, target_features)
    """

    def __init__(
        self,
        dataset,
    ) -> None:

        if not hasattr(
            dataset,
            "__len__",
        ):
            raise TypeError(
                "dataset must implement __len__."
            )

        if not hasattr(
            dataset,
            "__getitem__",
        ):
            raise TypeError(
                "dataset must implement __getitem__."
            )

        if len(dataset) <= 0:
            raise ValueError(
                "dataset cannot be empty."
            )

        self.dataset = dataset

    def __len__(
        self,
    ) -> int:

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        X, y = self.dataset[
            index
        ]

        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        y_tensor = torch.as_tensor(
            y,
            dtype=torch.float32,
        )

        if not torch.isfinite(
            X_tensor
        ).all():

            raise ValueError(
                "Input sample contains "
                "NaN or Inf."
            )

        if not torch.isfinite(
            y_tensor
        ).all():

            raise ValueError(
                "Target sample contains "
                "NaN or Inf."
            )

        return (
            X_tensor,
            y_tensor,
        )


# ============================================================
# DATA LOADERS
# ============================================================

def create_data_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    random_seed: int = 42,
) -> DataLoader:
    """
    Create a deterministic PyTorch DataLoader.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    torch_dataset = TorchForecastDataset(
        dataset
    )

    generator = torch.Generator()

    generator.manual_seed(
        random_seed
    )

    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

def build_loss_function(
    loss_name: str,
) -> nn.Module:
    """
    Create forecasting loss function.
    """

    loss_name = loss_name.lower()

    if loss_name == "mse":

        return nn.MSELoss()

    if loss_name == "mae":

        return nn.L1Loss()

    if loss_name == "huber":

        return nn.SmoothL1Loss()

    raise ValueError(
        f"Unsupported loss function "
        f"'{loss_name}'."
    )


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(
    model: nn.Module,
    config: ForecastTrainerConfig,
) -> torch.optim.Optimizer:
    """
    Build model optimizer.
    """

    optimizer_name = (
        config.optimizer.lower()
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(parameters) == 0:
        raise ValueError(
            "Model contains no trainable "
            "parameters."
        )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=(
                config.weight_decay
            ),
        )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=(
                config.weight_decay
            ),
        )

    raise ValueError(
        f"Unsupported optimizer "
        f"'{config.optimizer}'."
    )


# ============================================================
# SCHEDULER
# ============================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ForecastTrainerConfig,
):
    """
    Build optional learning-rate scheduler.
    """

    scheduler_name = (
        config.scheduler.lower()
    )

    if scheduler_name == "none":
        return None

    if scheduler_name == (
        "reduce_on_plateau"
    ):

        return (
            torch.optim.lr_scheduler
            .ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=(
                    config.scheduler_factor
                ),
                patience=(
                    config.scheduler_patience
                ),
                min_lr=(
                    config.minimum_learning_rate
                ),
            )
        )

    raise ValueError(
        f"Unsupported scheduler "
        f"'{config.scheduler}'."
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

@dataclass
class ForecastTrainingHistory:
    """
    Stores forecasting training history.
    """

    train_loss: List[float]
    validation_loss: List[float]
    learning_rate: List[float]

    best_epoch: Optional[int] = None
    best_validation_loss: float = float(
        "inf"
    )

    stopped_early: bool = False

    def to_dict(
        self,
    ) -> Dict[str, object]:

        return {
            "train_loss":
                list(
                    self.train_loss
                ),

            "validation_loss":
                list(
                    self.validation_loss
                ),

            "learning_rate":
                list(
                    self.learning_rate
                ),

            "best_epoch":
                self.best_epoch,

            "best_validation_loss":
                self.best_validation_loss,

            "stopped_early":
                self.stopped_early,
        }


# ============================================================
# TRAINER
# ============================================================

class ForecastTrainer:
    """
    Trainer for the reconstructed forecasting model.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[
            ForecastTrainerConfig
        ] = None,
    ) -> None:

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        if config is None:
            config = ForecastTrainerConfig()

        config.validate()

        self.config = config

        set_random_seed(
            config.random_seed
        )

        self.device = resolve_device(
            config.device
        )

        self.model = model.to(
            self.device
        )

        self.loss_function = (
            build_loss_function(
                config.loss_function
            )
        )

        self.optimizer = (
            build_optimizer(
                self.model,
                config,
            )
        )

        self.scheduler = (
            build_scheduler(
                self.optimizer,
                config,
            )
        )

        self.history = (
            ForecastTrainingHistory(
                train_loss=[],
                validation_loss=[],
                learning_rate=[],
            )
        )

        self.checkpoint_directory = Path(
            config.checkpoint_directory
        )

        self.checkpoint_path = (
            self.checkpoint_directory
            / config.checkpoint_filename
        )

    # ========================================================
    # SINGLE TRAINING EPOCH
    # ========================================================

    def train_epoch(
        self,
        data_loader: DataLoader,
    ) -> float:
        """
        Perform one training epoch.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for X, y in data_loader:

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            prediction = self.model(
                X
            )

            if prediction.shape != (
                y.shape
            ):
                raise ValueError(
                    "Prediction and target shapes "
                    "do not match. "
                    f"Prediction: "
                    f"{tuple(prediction.shape)}, "
                    f"target: {tuple(y.shape)}."
                )

            loss = self.loss_function(
                prediction,
                y,
            )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite training loss "
                    "encountered."
                )

            loss.backward()

            if (
                self.config
                .gradient_clip_norm
                is not None
            ):

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=(
                        self.config
                        .gradient_clip_norm
                    ),
                )

            self.optimizer.step()

            batch_size = X.shape[
                0
            ]

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_size
            )

            total_samples += (
                batch_size
            )

        if total_samples == 0:
            raise ValueError(
                "Training loader produced "
                "zero samples."
            )

        return (
            total_loss
            / total_samples
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_epoch(
        self,
        data_loader: DataLoader,
    ) -> float:
        """
        Evaluate one validation epoch.
        """

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for X, y in data_loader:

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            prediction = self.model(
                X
            )

            if prediction.shape != (
                y.shape
            ):
                raise ValueError(
                    "Prediction and target shapes "
                    "do not match during "
                    "validation."
                )

            loss = self.loss_function(
                prediction,
                y,
            )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite validation loss "
                    "encountered."
                )

            batch_size = X.shape[
                0
            ]

            total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            total_samples += (
                batch_size
            )

        if total_samples == 0:
            raise ValueError(
                "Validation loader produced "
                "zero samples."
            )

        return (
            total_loss
            / total_samples
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        epoch: int,
        validation_loss: float,
    ) -> Path:
        """
        Save current model and optimizer state.
        """

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch":
                int(
                    epoch
                ),

            "validation_loss":
                float(
                    validation_loss
                ),

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "trainer_config":
                asdict(
                    self.config
                ),

            "history":
                self.history.to_dict(),
        }

        if hasattr(
            self.model,
            "config",
        ):

            model_config = (
                getattr(
                    self.model,
                    "config",
                )
            )

            try:
                checkpoint[
                    "model_config"
                ] = asdict(
                    model_config
                )

            except TypeError:
                checkpoint[
                    "model_config"
                ] = None

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

        return self.checkpoint_path

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint_path: Optional[
            str | Path
        ] = None,
        load_optimizer: bool = True,
    ) -> Dict[str, object]:
        """
        Load model checkpoint.
        """

        if checkpoint_path is None:

            checkpoint_path = (
                self.checkpoint_path
            )

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

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            load_optimizer
            and "optimizer_state_dict"
            in checkpoint
        ):

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        return checkpoint

    # ========================================================
    # TRAIN COMPLETE MODEL
    # ========================================================

    def fit(
        self,
        train_dataset,
        validation_dataset,
        verbose: bool = True,
    ) -> ForecastTrainingHistory:
        """
        Train forecasting model using validation-based early stopping.
        """

        train_loader = (
            create_data_loader(
                dataset=train_dataset,
                batch_size=(
                    self.config.batch_size
                ),
                shuffle=True,
                random_seed=(
                    self.config.random_seed
                ),
            )
        )

        validation_loader = (
            create_data_loader(
                dataset=validation_dataset,
                batch_size=(
                    self.config.batch_size
                ),
                shuffle=False,
                random_seed=(
                    self.config.random_seed
                ),
            )
        )

        best_validation_loss = float(
            "inf"
        )

        best_epoch = None

        epochs_without_improvement = 0

        for epoch in range(
            1,
            self.config.maximum_epochs
            + 1,
        ):

            train_loss = self.train_epoch(
                train_loader
            )

            validation_loss = (
                self.validate_epoch(
                    validation_loader
                )
            )

            current_lr = (
                self.optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            self.history.train_loss.append(
                float(
                    train_loss
                )
            )

            self.history.validation_loss.append(
                float(
                    validation_loss
                )
            )

            self.history.learning_rate.append(
                float(
                    current_lr
                )
            )

            improvement = (
                best_validation_loss
                - validation_loss
            )

            if (
                improvement
                > self.config
                .minimum_improvement
            ):

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                self.history.best_epoch = (
                    epoch
                )

                self.history.best_validation_loss = (
                    float(
                        validation_loss
                    )
                )

                self.save_checkpoint(
                    epoch=epoch,
                    validation_loss=(
                        validation_loss
                    ),
                )

            else:

                epochs_without_improvement += 1

            if self.scheduler is not None:

                self.scheduler.step(
                    validation_loss
                )

            if verbose:

                print(
                    f"Epoch "
                    f"{epoch:03d}/"
                    f"{self.config.maximum_epochs:03d} | "
                    f"Train Loss: "
                    f"{train_loss:.8f} | "
                    f"Validation Loss: "
                    f"{validation_loss:.8f} | "
                    f"LR: {current_lr:.8f}"
                )

            patience = (
                self.config
                .early_stopping_patience
            )

            if (
                patience > 0
                and epochs_without_improvement
                >= patience
            ):

                self.history.stopped_early = (
                    True
                )

                if verbose:

                    print(
                        "Early stopping triggered "
                        f"at epoch {epoch}."
                    )

                break

        if best_epoch is None:

            raise RuntimeError(
                "Training completed without a "
                "valid best checkpoint."
            )

        # Restore the model weights associated with the
        # best validation epoch.
        self.load_checkpoint(
            self.checkpoint_path,
            load_optimizer=False,
        )

        return self.history

    # ========================================================
    # HISTORY SAVE
    # ========================================================

    def save_history(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save training history to JSON.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history.to_dict(),
                file,
                indent=4,
            )

        return path


# ============================================================
# HIGH-LEVEL TRAINING HELPER
# ============================================================

def train_forecasting_model(
    model: nn.Module,
    train_dataset,
    validation_dataset,
    config: Optional[
        ForecastTrainerConfig
    ] = None,
    verbose: bool = True,
) -> Tuple[
    nn.Module,
    ForecastTrainingHistory,
]:
    """
    Convenience function for complete model training.
    """

    trainer = ForecastTrainer(
        model=model,
        config=config,
    )

    history = trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        verbose=verbose,
    )

    return (
        trainer.model,
        history,
    )