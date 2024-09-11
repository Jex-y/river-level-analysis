import math
import re
from typing import Iterable, Literal

import torch
from einops import rearrange, reduce
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from .config import ActivationFunction, Config, Norm, PreprocessingType
from .preprocessing import (
    IdentityPreprocessing,
    QuantilePreprocessing,
    StandardPreprocessing,
)


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


def conditional_layer(layer, condition):
    return layer if condition else nn.Identity()


def get_preprocessing_layer(
    feature_type: Literal["level", "rainfall", "y"], config: Config
):
    match feature_type:
        case "level":
            preprocessing_type = config.level_preprocessing
        case "rainfall":
            preprocessing_type = config.rainfall_preprocessing
        case "y":
            preprocessing_type = config.y_preprocessing

    match preprocessing_type:
        case PreprocessingType.STANDARD:
            return StandardPreprocessing()
        case PreprocessingType.QUANTILE:
            return QuantilePreprocessing(config)
        case PreprocessingType.MINMAX:
            raise NotImplementedError
        case PreprocessingType.NONE:
            return IdentityPreprocessing()


class BaseTimeSeriesModel(LightningModule):
    # Buffers
    level_cols: torch.BoolTensor
    rainfall_cols: torch.BoolTensor

    def __init__(self, input_column_names: Iterable[str], config: Config):
        super().__init__()
        self.config = config
        self.required_samples = max(*config.rolling_windows, config.context_length)
        self.n_context_features = len(input_column_names)
        self.x_column_names = input_column_names

        self.target_feature_index = list(input_column_names).index(config.target_col)

        self.level_preprocessing = get_preprocessing_layer("level", self.config)
        self.rainfall_preprocessing = get_preprocessing_layer("rainfall", self.config)
        self.level_rolling_preprocessing = get_preprocessing_layer("level", self.config)
        self.rainfall_rolling_preprocessing = get_preprocessing_layer(
            "rainfall", self.config
        )
        self.y_preprocessing = get_preprocessing_layer("y", self.config)

    def forward(
        self,
        features_with_time_dim: torch.FloatTensor,
        features_without_time_dim: torch.FloatTensor,
    ):
        """Forward pass of the model.

        Args:
            features_with_time (torch.FloatTensor): Shape (batch_size, context_length, -1)
            features_without_time (torch.FloatTensor): Shape (batch_size, -1)
        """
        pass

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
        context: torch.FloatTensor,
        return_logits: bool = False,
    ):
        batch_size = context.shape[0]

        datetime_features = self.calculate_timestamp_features(time)
        # Shape should be (batch_size, 3)

        rolling_features = self.calculate_rolling_features(context)
        # Shape should be (batch_size, len(rolling_windows) * num_features)

        context = context[:, -self.config.context_length :]
        # Shape should be (batch_size, context_length, num_features)

        # Apply preprocessing

        context[:, :, self.level_cols] = self.level_preprocessing.transform(
            context[:, :, self.level_cols]
        )
        context[:, :, self.rainfall_cols] = self.rainfall_preprocessing.transform(
            context[:, :, self.rainfall_cols]
        )

        rolling_level_cols = self.level_cols.repeat(len(self.config.rolling_windows))
        rolling_rainfall_cols = self.rainfall_cols.repeat(
            len(self.config.rolling_windows)
        )

        rolling_features[:, rolling_level_cols] = (
            self.level_rolling_preprocessing.transform(
                rolling_features[:, rolling_level_cols]
            )
        )

        rolling_features[:, rolling_rainfall_cols] = (
            self.rainfall_rolling_preprocessing.transform(
                rolling_features[:, rolling_rainfall_cols]
            )
        )

        # mean, quantiles, thresholds_logits
        last_dim_splits = [1, len(self.config.quantiles), len(self.config.thresholds)]

        # output = self.forward(
        #     torch.cat([context.flatten(1), time_features, rolling_features], dim=-1)
        # ).view(batch_size, self.config.prediction_length, sum(last_dim_splits))
        output = self.forward(
            context,
            torch.cat([datetime_features, rolling_features], dim=-1),
        ).view(batch_size, self.config.prediction_length, sum(last_dim_splits))

        pred_mean_transformed, pred_quantiles_transformed, pred_thresholds_logits = (
            torch.split(
                output,
                last_dim_splits,
                dim=-1,
            )
        )

        pred_mean = self.y_preprocessing.inverse_transform(pred_mean_transformed)

        pred_quantiles = rearrange(
            self.y_preprocessing.inverse_transform(
                rearrange(pred_quantiles_transformed, "bs np nq -> (bs np nq) 1")
            ),
            "(bs np nq) 1 -> bs np nq",
            bs=batch_size,
            np=self.config.prediction_length,
            nq=len(self.config.quantiles),
        )

        if not return_logits:
            pred_thresholds_logits = pred_thresholds_logits.sigmoid()

        return pred_mean, pred_quantiles, pred_thresholds_logits

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

    def fit_preprocessing(self, x: torch.Tensor, y: torch.Tensor):
        level_col_regex = re.compile(r"level$")
        rainfall_col_regex = re.compile(r"rainfall$")

        level_cols = torch.tensor(
            [bool(level_col_regex.search(col)) for col in self.x_column_names],
            device=x.device,
        )

        rainfall_cols = torch.tensor(
            [bool(rainfall_col_regex.search(col)) for col in self.x_column_names],
            device=x.device,
        )

        self.register_buffer("level_cols", level_cols)
        self.register_buffer("rainfall_cols", rainfall_cols)

        self.level_preprocessing.fit(x[:, level_cols])
        self.rainfall_preprocessing.fit(x[:, rainfall_cols])

        # X is of shape (n_samples, n_features)

        # rolling features should be of shape (n_samples - longest_window + 1, n_features * n_windows)

        n_output_samples = x.shape[0] - max(*self.config.rolling_windows) + 1
        # Get only last n_output_samples for each window size to make sure they are all the same length

        # x.unfold(0, window, 1) creates a tensor of shape (n_samples - window + 1, n_features, window)
        # we then want to take the mean of the window, this is dim -1 of the unfolded tensor
        # Output should then be of shape (n_samples - window + 1, n_features)

        rolling_features = torch.cat(
            [
                x.unfold(0, window, 1).mean(dim=-1)[-n_output_samples:]
                for window in self.config.rolling_windows
            ],
            dim=-1,
        )
        # Last dimension should be (n_features at window 1, n_features at window 2, ...)

        self.level_rolling_preprocessing.fit(
            rolling_features[:, level_cols.repeat(len(self.config.rolling_windows))]
        )

        self.rainfall_rolling_preprocessing.fit(
            rolling_features[:, rainfall_cols.repeat(len(self.config.rolling_windows))]
        )

        self.y_preprocessing.fit(y)

        return self

    def calculate_metrics(
        self, batch: tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]
    ):
        x_datetime, x, y_true = batch
        pred_median, pred_quantiles, pred_thresholds = self.forecast(
            x_datetime, x, return_logits=True
        )
        # return logits to improve numerical stability

        quantiles = torch.tensor(self.config.quantiles, device=y_true.device)
        thresholds = torch.tensor(self.config.thresholds, device=y_true.device)

        # Mean Absolute Error loss

        mae_loss = (y_true - pred_median).abs().mean()

        # Quantile loss

        quantile_error = y_true - pred_quantiles

        quantile_loss = reduce(
            torch.max((quantiles - 1) * quantile_error, quantiles * quantile_error),
            "bs np nq -> nq",
            "mean",
        )

        # Threshold loss

        y_true_over_threshold = (y_true > thresholds).float()

        threshold_loss = reduce(
            binary_cross_entropy_with_logits(
                pred_thresholds,
                y_true_over_threshold,
                reduction="none",
            ),
            "bs np nt -> nt",
            "mean",
        )

        total_loss = (
            (quantile_loss.mean() * self.config.quantile_loss_coefficient)
            + (threshold_loss.mean() * self.config.threshold_loss_coefficient)
            + (mae_loss * self.config.mae_loss_coefficient)
        )

        y_pred_over_threhsold = torch.sigmoid(pred_thresholds) > 0.5

        return total_loss, {
            "total_loss": total_loss,
            "mae_loss": mae_loss,
            "total_quantile_loss": quantile_loss.mean(),
            "total_threshold_loss": threshold_loss.mean(),
            **{
                f"quantile_loss_{q}": quantile_loss[i]
                for i, q in enumerate(self.config.quantiles)
            },
            **{
                f"threshold_loss_{t}": threshold_loss[i]
                for i, t in enumerate(self.config.thresholds)
            },
            **{
                f"threshold_accuracy_{t}": (
                    y_true_over_threshold[i] == y_pred_over_threhsold[i]
                )
                .float()
                .mean()
                for i, t in enumerate(self.config.thresholds)
            },
            **{
                f"quantile_{q}_coverage": (y_true <= pred_quantiles[:, i : i + 1])
                .float()
                .mean()
                for i, q in enumerate(self.config.quantiles)
            },
        }

    def training_step(self, batch, _batch_idx):
        loss, metrics = self.calculate_metrics(batch)
        self.log_dict(
            add_prefix("train", metrics),
            on_epoch=False,
            on_step=True,
            # prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        _, metrics = self.calculate_metrics(batch)
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
                bias=not config.norm_before_activation and config.mlp_norm != Norm.NONE,
            ),
            conditional_layer(
                get_mlp_norm_layer(config, output_size),
                config.norm_before_activation,
            ),
            get_activation_function(config),
            conditional_layer(
                get_mlp_norm_layer(config, output_size),
                not config.norm_before_activation,
            ),
            get_dropout_layer(config) if not is_last else nn.Identity(),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_features, output_features, config: Config):
        super().__init__(
            nn.Conv1d(
                input_features,
                output_features,
                config.conv_kernel_size,
                padding=config.conv_kernel_size // 2,
                bias=not config.norm_before_activation
                and config.conv_norm != Norm.NONE,
            ),
            conditional_layer(
                get_conv_norm_layer(config, output_features),
                config.norm_before_activation,
            ),
            get_activation_function(config),
            conditional_layer(
                get_conv_norm_layer(config, output_features),
                not config.norm_before_activation,
            ),
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

        sizes = (
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
                    sizes[:-1], sizes[1:], is_last
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
