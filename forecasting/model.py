from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Optional
import torch
from torch import nn
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ForecastModelConfig:
    """
    Configuration of the multi-horizon forecasting model.
    """
    # Manuscript-supported dimensions
    input_window: int = 168
    forecast_horizon: int = 24

    # Number of operational variables.
    input_features: int = 4
    target_features: int = 4

    # Reconstruction choices
    hidden_dimension: int = 128
    number_of_attention_heads: int = 4
    number_of_encoder_layers: int = 2
    feedforward_dimension: int = 256
    dropout: float = 0.10

    activation: str = "gelu"

    use_learnable_positional_encoding: bool = True

    def validate(self) -> None:
        """Validate forecasting-model configuration."""

        integer_parameters = {
            "input_window": self.input_window,
            "forecast_horizon": self.forecast_horizon,
            "input_features": self.input_features,
            "target_features": self.target_features,
            "hidden_dimension": self.hidden_dimension,
            "number_of_attention_heads":
                self.number_of_attention_heads,
            "number_of_encoder_layers":
                self.number_of_encoder_layers,
            "feedforward_dimension":
                self.feedforward_dimension,
        }

        for name, value in integer_parameters.items():

            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        if (
            self.hidden_dimension
            % self.number_of_attention_heads
            != 0
        ):
            raise ValueError(
                "hidden_dimension must be divisible by "
                "number_of_attention_heads."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        supported_activations = {
            "relu",
            "gelu",
        }

        if self.activation not in supported_activations:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Supported values are "
                f"{sorted(supported_activations)}."
            )
# ============================================================
# POSITIONAL ENCODING
# ============================================================

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional representation for hourly observations.

    Shape
    -----
    Input:
        (batch, sequence_length, hidden_dimension)

    Output:
        same shape
    """

    def __init__(
        self,
        maximum_sequence_length: int,
        hidden_dimension: int,
    ) -> None:

        super().__init__()

        if maximum_sequence_length <= 0:
            raise ValueError(
                "maximum_sequence_length must be positive."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.hidden_dimension = (
            hidden_dimension
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                maximum_sequence_length,
                hidden_dimension,
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Add learned positional information."""

        if x.ndim != 3:
            raise ValueError(
                "Positional encoding expects a 3-D tensor "
                "(batch, sequence, features)."
            )

        sequence_length = x.shape[
            1
        ]

        if sequence_length > (
            self.maximum_sequence_length
        ):
            raise ValueError(
                "Input sequence is longer than the configured "
                "maximum sequence length."
            )

        return (
            x
            + self.position_embedding[
                :,
                :sequence_length,
                :
            ]
        )


# ============================================================
# FIXED SINUSOIDAL POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Deterministic sinusoidal positional encoding.

    Provided as an alternative when learnable positional encoding
    is disabled.
    """

    def __init__(
        self,
        maximum_sequence_length: int,
        hidden_dimension: int,
    ) -> None:

        super().__init__()

        if maximum_sequence_length <= 0:
            raise ValueError(
                "maximum_sequence_length must be positive."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        position = torch.arange(
            maximum_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        dimension = torch.arange(
            0,
            hidden_dimension,
            2,
            dtype=torch.float32,
        )

        denominator = torch.exp(
            dimension
            * (
                -torch.log(
                    torch.tensor(
                        10000.0
                    )
                )
                / hidden_dimension
            )
        )

        encoding = torch.zeros(
            maximum_sequence_length,
            hidden_dimension,
            dtype=torch.float32,
        )

        encoding[
            :,
            0::2
        ] = torch.sin(
            position
            * denominator
        )

        # Handle odd hidden dimensions safely.
        odd_width = encoding[
            :,
            1::2
        ].shape[
            1
        ]

        encoding[
            :,
            1::2
        ] = torch.cos(
            position
            * denominator[
                :odd_width
            ]
        )

        self.register_buffer(
            "position_encoding",
            encoding.unsqueeze(
                0
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Positional encoding expects a 3-D tensor."
            )

        sequence_length = x.shape[
            1
        ]

        if sequence_length > (
            self.position_encoding.shape[
                1
            ]
        ):
            raise ValueError(
                "Input sequence exceeds maximum length."
            )

        return (
            x
            + self.position_encoding[
                :,
                :sequence_length,
                :
            ]
        )


# ============================================================
# MAIN FORECASTING MODEL
# ============================================================

class MultiHorizonTransformerForecaster(
    nn.Module
):
    """
    Direct multi-horizon transformer forecasting model.

    Input
    -----
    x:
        shape:
            (batch_size,
             input_window,
             input_features)

    Output
    ------
    forecast:
        shape:
            (batch_size,
             forecast_horizon,
             target_features)
    """

    def __init__(
        self,
        config: Optional[
            ForecastModelConfig
        ] = None,
    ) -> None:

        super().__init__()

        if config is None:
            config = ForecastModelConfig()

        config.validate()

        self.config = config

        # ----------------------------------------------------
        # INPUT EMBEDDING
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            config.input_features,
            config.hidden_dimension,
        )

        # ----------------------------------------------------
        # POSITIONAL ENCODING
        # ----------------------------------------------------

        if (
            config.use_learnable_positional_encoding
        ):

            self.positional_encoding = (
                LearnablePositionalEncoding(
                    maximum_sequence_length=(
                        config.input_window
                    ),
                    hidden_dimension=(
                        config.hidden_dimension
                    ),
                )
            )

        else:

            self.positional_encoding = (
                SinusoidalPositionalEncoding(
                    maximum_sequence_length=(
                        config.input_window
                    ),
                    hidden_dimension=(
                        config.hidden_dimension
                    ),
                )
            )

        # ----------------------------------------------------
        # TRANSFORMER ENCODER
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=(
                config.hidden_dimension
            ),
            nhead=(
                config.number_of_attention_heads
            ),
            dim_feedforward=(
                config.feedforward_dimension
            ),
            dropout=(
                config.dropout
            ),
            activation=(
                config.activation
            ),
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=(
                    config.number_of_encoder_layers
                ),
            )
        )

        # ----------------------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------------------

        self.output_norm = nn.LayerNorm(
            config.hidden_dimension
        )

        # ----------------------------------------------------
        # TEMPORAL ATTENTION POOLING
        # ----------------------------------------------------
        #
        # Instead of using only the final hour, the model learns
        # which historical time steps are most important.
        #
        # This produces one global temporal representation:
        #
        #     (batch, hidden_dimension)
        # ----------------------------------------------------

        self.temporal_attention = nn.Linear(
            config.hidden_dimension,
            1,
        )

        # ----------------------------------------------------
        # MULTI-HORIZON OUTPUT HEAD
        # ----------------------------------------------------

        output_dimension = (
            config.forecast_horizon
            * config.target_features
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(
                config.hidden_dimension,
                config.hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_dimension,
                output_dimension,
            ),
        )

        self._initialize_weights()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(
        self,
    ) -> None:
        """
        Initialize linear layers using Xavier initialization.
        """

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                "Model input must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Model input must have shape "
                "(batch, sequence, features)."
            )

        if x.shape[
            1
        ] != self.config.input_window:

            raise ValueError(
                "Incorrect input sequence length. "
                f"Expected {self.config.input_window}, "
                f"received {x.shape[1]}."
            )

        if x.shape[
            2
        ] != self.config.input_features:

            raise ValueError(
                "Incorrect number of input features. "
                f"Expected {self.config.input_features}, "
                f"received {x.shape[2]}."
            )

        if not torch.isfinite(
            x
        ).all():

            raise ValueError(
                "Model input contains NaN or Inf values."
            )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the 168-hour historical sequence.

        Returns
        -------
        Tensor
            Shape:
                (batch,
                 input_window,
                 hidden_dimension)
        """

        self._validate_input(
            x
        )

        hidden = self.input_projection(
            x
        )

        hidden = self.positional_encoding(
            hidden
        )

        hidden = self.transformer_encoder(
            hidden
        )

        hidden = self.output_norm(
            hidden
        )

        return hidden

    # ========================================================
    # TEMPORAL POOLING
    # ========================================================

    def temporal_pool(
        self,
        encoded: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Attention-based pooling across historical time steps.

        Returns
        -------
        context:
            (batch, hidden_dimension)

        attention_weights:
            (batch, input_window)
        """

        if encoded.ndim != 3:
            raise ValueError(
                "Encoded representation must be 3-D."
            )

        attention_logits = (
            self.temporal_attention(
                encoded
            ).squeeze(
                -1
            )
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        context = torch.sum(
            encoded
            * attention_weights.unsqueeze(
                -1
            ),
            dim=1,
        )

        return (
            context,
            attention_weights,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Produce direct 24-hour forecasts.

        Parameters
        ----------
        x:
            Input tensor:

                (batch,
                 168,
                 input_features)

        return_attention:
            When True, also return temporal attention weights.

        Returns
        -------
        forecast:
            (batch,
             24,
             target_features)

        or

        forecast, attention_weights
        """

        encoded = self.encode(
            x
        )

        (
            context,
            attention_weights,
        ) = self.temporal_pool(
            encoded
        )

        flat_forecast = self.forecast_head(
            context
        )

        forecast = flat_forecast.reshape(
            x.shape[
                0
            ],
            self.config.forecast_horizon,
            self.config.target_features,
        )

        if return_attention:

            return (
                forecast,
                attention_weights,
            )

        return forecast

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def number_of_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        """
        Count model parameters.
        """

        if trainable_only:

            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def model_summary(
        self,
    ) -> Dict[str, object]:
        """
        Return architecture metadata for reproducibility.
        """

        return {
            "model_class":
                self.__class__.__name__,

            "configuration":
                asdict(
                    self.config
                ),

            "trainable_parameters":
                self.number_of_parameters(
                    trainable_only=True
                ),

            "total_parameters":
                self.number_of_parameters(
                    trainable_only=False
                ),

            "expected_input_shape": (
                None,
                self.config.input_window,
                self.config.input_features,
            ),

            "expected_output_shape": (
                None,
                self.config.forecast_horizon,
                self.config.target_features,
            ),
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "MultiHorizonTransformerForecaster("
            f"input_window={self.config.input_window}, "
            f"horizon={self.config.forecast_horizon}, "
            f"input_features={self.config.input_features}, "
            f"target_features={self.config.target_features}, "
            f"hidden={self.config.hidden_dimension}, "
            f"heads={self.config.number_of_attention_heads}, "
            f"layers={self.config.number_of_encoder_layers}"
            ")"
        )


# ============================================================
# FACTORY
# ============================================================

def build_forecasting_model(
    input_features: int = 4,
    target_features: int = 4,
    input_window: int = 168,
    forecast_horizon: int = 24,
    hidden_dimension: int = 128,
    number_of_attention_heads: int = 4,
    number_of_encoder_layers: int = 2,
    feedforward_dimension: int = 256,
    dropout: float = 0.10,
    activation: str = "gelu",
    use_learnable_positional_encoding: bool = True,
) -> MultiHorizonTransformerForecaster:
    """
    Convenience factory for the FC-HMARL forecasting model.
    """

    config = ForecastModelConfig(
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        input_features=input_features,
        target_features=target_features,
        hidden_dimension=hidden_dimension,
        number_of_attention_heads=(
            number_of_attention_heads
        ),
        number_of_encoder_layers=(
            number_of_encoder_layers
        ),
        feedforward_dimension=(
            feedforward_dimension
        ),
        dropout=dropout,
        activation=activation,
        use_learnable_positional_encoding=(
            use_learnable_positional_encoding
        ),
    )

    return MultiHorizonTransformerForecaster(
        config
    )
