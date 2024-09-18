import math
import re
from typing import Iterable, Literal

import torch
from einops import rearrange, reduce
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, gaussian_nll_loss

from .config import ActivationFunction, Config, Norm, PreprocessingType

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
    level_cols: torch.BoolTensor
    rainfall_cols: torch.BoolTensor

    def __init__(self, input_column_names: list[str], config: Config):
        super().__init__()
        self.config = config
        self.required_samples = max(*config.rolling_windows, config.context_length)
        self.n_context_features = len(input_column_names)
        self.x_column_names = input_column_names

        self.target_feature_index = list(input_column_names).index(config.target_col)

    def forward(
        self,
        features_with_time_dim: torch.FloatTensor,
        features_without_time_dim: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Forward pass of the model.

        Args:
            features_with_time (torch.FloatTensor): Shape (batch_size, context_length, -1)
            features_without_time (torch.FloatTensor): Shape (batch_size, -1)
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
        return (2 + len(self.config.thresholds)) * self.config.prediction_length

    def forecast(
        self,
        time: torch.LongTensor,
        context: torch.FloatTensor,
        return_raw: bool = False,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_size = context.shape[0]

        datetime_features = self.calculate_timestamp_features(time)
        # Shape should be (batch_size, 3)

        rolling_features = self.calculate_rolling_features(context)
        # Shape should be (batch_size, len(rolling_windows) * num_features)

        context = context[:, -self.config.context_length :]
        # Shape should be (batch_size, context_length, num_features)


        # mean, quantiles, thresholds_logits
        last_dim_splits = [1, 1, len(self.config.thresholds)]

        output = self.forward(
            context,
            torch.cat([datetime_features, rolling_features], dim=-1),
        ).view(batch_size, self.config.prediction_length, sum(last_dim_splits))

        pred_mean, pred_log_variance, pred_thresholds_logits = (
            torch.split(
                output,
                last_dim_splits,
                dim=-1,
            )
        )

        if return_raw:
            return (
                pred_mean,
                pred_log_variance,
                pred_thresholds_logits,
            )

        # To deal with variance and quantile transform, could find +/-1std, +/-2std, ect, inverse quantile transform these and return them as well.
        pred_stddev = (
            0.5 * pred_log_variance
        ).exp()

        return pred_mean, pred_stddev, pred_thresholds_logits.sigmoid()

    def calculate_timestamp_features(self, time: torch.LongTensor) -> torch.FloatTensor:
        # time is expected to be of shape (batch_size, 2)

        day_of_year, year = time.unbind(dim=-1)

        day_of_year = day_of_year.type(torch.float32) / 365.25
        sin_day_of_year = (day_of_year * 2 * math.pi).sin()
        cos_day_of_year = (day_of_year * 2 * math.pi).cos()

        years_since_2007 = (year - 2007).type(torch.float32)

        return torch.stack([sin_day_of_year, cos_day_of_year, years_since_2007], dim=-1)

    def calculate_rolling_features(
        self,
        features_to_roll: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
        batch: tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor],
        return_metrics: bool = False,
    ):
        x_datetime, x, y_true = batch
        (
            pred_mean,
            pred_log_variance,
            pred_thresholds_logits,
        ) = self.forecast(x_datetime, x, return_raw=True)
        # return logits to improve numerical stability

        thresholds = torch.tensor(self.config.thresholds, device=y_true.device)

        # Mean Absolute Error loss

        # NNL Loss
        squared_error = (pred_mean - y_true) ** 2


        nnl_loss = reduce(
            0.5 * (pred_log_variance + (squared_error / pred_log_variance.exp().clamp(min=1e-6)) + math.log(2 * math.pi)),
            "bs np 1 -> np",
            "mean",
        )

        # Threshold Loss

        y_true_over_threshold = (y_true > thresholds).float()

        threshold_loss = reduce(
            binary_cross_entropy_with_logits(
                pred_thresholds_logits,
                y_true_over_threshold,
                reduction="none",
            ),
            "bs np nt -> np nt",
            "mean",
        )

        total_loss = (nnl_loss.mean() * self.config.nnl_loss_coefficient) + (
            threshold_loss.mean() * self.config.threshold_loss_coefficient
        )

        if not return_metrics:
            return total_loss

        pred_stddev = (
            0.5 * pred_log_variance
        ).exp()

        return total_loss, {
            "total_loss": total_loss,
            "total_nnl_loss": nnl_loss.mean(),
            "total_threshold_loss": threshold_loss.mean(),
            "mse_transformed": squared_error.mean(),
            "mse_true_value": torch.square(pred_mean - y_true).mean(),
            "threshold_accuracy": (pred_thresholds_logits.sigmoid() > 0.5).eq(y_true_over_threshold).float().mean(),
            **{
                f"{stdevs} stdev coverage": (y_true > (pred_mean - stdevs * pred_stddev)).float().mean()
                for stdevs in (1, 2)
            }
            # "nnl_loss": nnl_loss,
            # "threshold_loss": threshold_loss,
            # "threshold_accuracy": threshold_accuracy,
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
                padding='same',
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
