import math
from typing import Iterable

import torch
from einops import reduce
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F

from .config import ActivationFunction, Config, Norm


def smooth_pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    alpha: float = 0.001,
):
    error = target - pred
    q_error = quantiles * error
    beta = 1 / alpha
    soft_error = F.softplus(-error, beta)
    losses = q_error + soft_error
    return losses.mean(dim=1).sum()


def add_prefix(prefix, dictionary):
    return {f"{prefix}_{k}": v for k, v in dictionary.items()}


def get_activation_function(config: Config):
    match config.activation_function:
        case ActivationFunction.RELU:
            return nn.ReLU()
        case ActivationFunction.GELU:
            return nn.GELU()
        case ActivationFunction.TANH:
            return nn.Tanh()
        case ActivationFunction.ELU:
            return nn.ELU()
        case ActivationFunction.SWISH:
            return nn.SiLU()
        case _:
            raise ValueError(
                f"Invalid activation function: {config.activation_function}."
            )


def get_mlp_norm_layer(config: Config, size: int):
    match config.mlp_norm:
        case Norm.BATCH:
            return nn.BatchNorm1d(size)
        case Norm.LAYER:
            return nn.LayerNorm(size)
        case Norm.NONE:
            return nn.Identity()
        case _:
            raise ValueError(f"Invalid norm layer: {config.norm}.")


def get_conv_norm_layer(config: Config, features: int):
    match config.conv_norm:
        case Norm.BATCH:
            return nn.BatchNorm1d(features)
        case Norm.LAYER:
            return nn.LayerNorm((features, config.context_length))
        case Norm.NONE:
            return nn.Identity()
        case _:
            raise ValueError(f"Invalid norm layer: {config.norm}.")


def get_dropout_layer(config: Config):
    return nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()


class BaseTimeSeriesModel(LightningModule):
    # Buffers
    thresholds: torch.Tensor
    quantiles: torch.Tensor

    def __init__(self, input_column_names: list[str], config: Config):
        super().__init__()
        self.config = config
        self.required_samples = max(*config.rolling_windows, config.context_length)
        self.n_context_features = len(input_column_names)
        self.x_column_names = input_column_names

        self.quantiles = torch.tensor(
            config.quantiles, dtype=torch.float32, device=self.device
        )

        self.register_buffer(
            "thresholds",
            torch.tensor(config.thresholds, dtype=torch.float32, device=self.device),
        )

    def forward(
        self,
        features_with_time_dim: torch.Tensor,
        features_without_time_dim: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            features_with_time (torch.Tensor): Shape (batch_size, context_length, -1)
            features_without_time (torch.Tensor): Shape (batch_size, -1)
        """
        pass

    def get_example_forecast_input(self, bs=1, return_raw=False):
        return (
            torch.zeros(bs, 2, dtype=torch.long),
            torch.zeros(bs, self.required_samples, self.n_context_features),
            return_raw,
        )

    @property
    def num_time_features(self):
        return self.n_context_features

    @property
    def num_non_time_features(self):
        return (len(self.config.rolling_windows) * self.n_context_features) + 3

    @property
    def num_output_features(self):
        return (
            1 + len(self.config.quantiles) + len(self.config.thresholds)
        ) * self.config.prediction_length

    def forecast(
        self,
        time: torch.LongTensor,
        context: torch.Tensor,
        return_raw: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]

        datetime_features = self.calculate_timestamp_features(time)
        # Shape should be (batch_size, 3)

        rolling_features = self.calculate_rolling_features(context)
        # Shape should be (batch_size, len(rolling_windows) * num_features)

        context = context[:, -self.config.context_length :]
        # Shape should be (batch_size, context_length, num_features)

        # mean, quantiles, thresholds_logits
        last_dim_splits = [1, len(self.config.quantiles), len(self.config.thresholds)]

        output = self.forward(
            context,
            torch.cat([datetime_features, rolling_features], dim=-1),
        ).view(batch_size, self.config.prediction_length, sum(last_dim_splits))

        pred_mean, pred_quantiles, pred_thresholds_logits = torch.split(
            output,
            last_dim_splits,
            dim=-1,
        )

        pred_quantiles = pred_quantiles + pred_mean

        if return_raw:
            return (
                pred_mean,
                pred_quantiles,
                pred_thresholds_logits,
            )

        return (
            pred_mean,
            pred_quantiles,
            pred_thresholds_logits.sigmoid(),
        )

    def calculate_timestamp_features(self, time: torch.LongTensor) -> torch.Tensor:
        # time is expected to be of shape (batch_size, 2)

        day_of_year, year = time.unbind(dim=-1)

        day_of_year = day_of_year.type(torch.float32) / 365.25
        sin_day_of_year = (day_of_year * 2 * math.pi).sin()
        cos_day_of_year = (day_of_year * 2 * math.pi).cos()

        years_since_2007 = (year - 2007).type(torch.float32)

        return torch.stack([sin_day_of_year, cos_day_of_year, years_since_2007], dim=-1)

    def calculate_rolling_features(
        self,
        features_to_roll: torch.Tensor,
    ) -> torch.Tensor:
        # features_to_roll is expected to be of shape (batch_size, n_timesteps, n_features)
        # We want to output a tensor of shape (batch_size, len(rolling_windows), n_features)
        # Calculates only the last window of each rolling window as required for inference.

        return torch.cat(
            [
                features_to_roll[:, -window:].mean(dim=1)
                for window in self.config.rolling_windows
            ],
            dim=1,
        )

    def calculate_metrics(
        self,
        batch: tuple[torch.LongTensor, torch.Tensor, torch.Tensor],
        return_metrics: bool = False,
    ):
        x_datetime, x, y_true = batch
        (
            pred_mean,
            pred_quantiles,
            pred_thresholds_logits,
        ) = self.forecast(x_datetime, x, return_raw=True)
        # return logits to improve numerical stability

        mse_loss = torch.square(pred_mean - y_true).mean()
        quantile_loss = smooth_pinball_loss(
            pred_quantiles,
            y_true,
            self.quantiles,
        )

        # Threshold Loss
        y_true_over_threshold = (y_true > self.thresholds).float()
        threshold_loss = binary_cross_entropy_with_logits(
            pred_thresholds_logits, y_true_over_threshold
        ).mean()

        total_loss = mse_loss + quantile_loss + threshold_loss

        if not return_metrics:
            return total_loss

        return total_loss, {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "threshold_loss": threshold_loss,
            "quantile_loss": quantile_loss,
            "threshold_accuracy": (pred_thresholds_logits.sigmoid() > 0.5)
            .eq(y_true_over_threshold)
            .float()
            .mean(),
            **{
                f"{q:.3f} quantile coverage": (y_true > pred_quantiles[..., i : i + 1])
                .float()
                .mean()
                for i, q in enumerate(self.config.quantiles)
            },
        }

    def training_step(self, batch, _batch_idx):
        loss, metrics = self.calculate_metrics(batch, return_metrics=True)
        self.log_dict(
            add_prefix("train", metrics),
            on_epoch=False,
            on_step=True,
            # prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        _, metrics = self.calculate_metrics(batch, return_metrics=True)
        self.log_dict(
            add_prefix("val", metrics),
            on_epoch=True,
            on_step=False,
            # prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            # fused=True,
            weight_decay=self.config.weight_decay,
        )
        return optimizer


class MLPBlock(nn.Sequential):
    def __init__(
        self,
        input_size,
        output_size,
        config: Config,
        is_last: bool = False,
    ):
        super().__init__(
            nn.Linear(
                input_size,
                output_size,
            ),
            get_activation_function(config),
            get_mlp_norm_layer(config, output_size),
            get_dropout_layer(config) if not is_last else nn.Identity(),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_features, output_features, config: Config):
        super().__init__(
            nn.Conv1d(
                input_features,
                output_features,
                config.conv_kernel_size,
                padding="same",
            ),
            get_activation_function(config),
            get_conv_norm_layer(config, output_features),
            get_dropout_layer(config),
        )


class TimeSeriesModel(BaseTimeSeriesModel):
    def __init__(
        self,
        input_column_names: Iterable[str],
        config: Config,
    ):
        super().__init__(input_column_names, config)

        # Conv layers are applied to features with a time dimension
        # These features are then flattened and passed through the MLP layers

        conv_sizes = [self.num_time_features] + [
            config.conv_hidden_size
        ] * config.num_conv_blocks

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(
                    input_features=input_features,
                    output_features=output_features,
                    config=config,
                )
                for input_features, output_features in zip(
                    conv_sizes[:-1], conv_sizes[1:]
                )
            ]
        )

        mlp_input_size = (
            self.num_non_time_features
            + conv_sizes[-1] * self.config.context_length
            + (
                self.num_time_features * self.config.context_length
                if config.skip_connection
                else 0
            )
        )

        mlp_sizes = (
            [mlp_input_size]
            + [config.mlp_hidden_size] * config.num_mlp_blocks
            + [self.num_output_features]
        )

        is_last = [False] * config.num_mlp_blocks + [True]

        self.mlp_layers = nn.Sequential(
            *[
                MLPBlock(
                    input_size=input_size,
                    output_size=output_size,
                    is_last=is_last,
                    config=config,
                )
                for input_size, output_size, is_last in zip(
                    mlp_sizes[:-1], mlp_sizes[1:], is_last
                )
            ],
        )

    def forward(
        self, features_with_time: torch.Tensor, features_without_time: torch.Tensor
    ):
        features_with_time_after_conv = self.conv_layers(
            features_with_time.permute(0, 2, 1)
        ).permute(0, 2, 1)

        if self.config.skip_connection:
            features_with_time_after_conv = torch.cat(
                [features_with_time, features_with_time_after_conv], dim=-1
            )

        return self.mlp_layers(
            torch.cat(
                [features_with_time_after_conv.flatten(1), features_without_time],
                dim=-1,
            )
        )
