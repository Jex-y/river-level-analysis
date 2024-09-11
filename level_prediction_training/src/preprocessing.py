import math

import torch
from einops import rearrange, repeat
from torch import nn

from src.config import Config


class PreprocessingTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, x: torch.Tensor):
        # Fit the preprocessing transform
        self.fitted = True
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the preprocessing transform
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the inverse preprocessing transform
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Preprocessing transform must be fitted before use")

        return self.transform(x).detach()


class IdentityPreprocessing(PreprocessingTransform):
    def __init__(self):
        super().__init()

    def fit(self, x: torch.Tensor):
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StandardPreprocessing(PreprocessingTransform):
    def __init__(self):
        super().__init()

    def fit(self, x: torch.Tensor):
        super().fit(x)
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class QuantilePreprocessing(PreprocessingTransform):
    quantiles: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.n_quantiles = config.quantile_preprocessing_n_quantiles
        self.output_normal = config.quantile_preprocessing_output_normal

    def fit(self, x):
        super().fit(x)
        self.register_buffer(
            "quantiles",
            torch.quantile(
                x, torch.linspace(0, 1, self.n_quantiles, device=x.device), dim=0
            ),
        )
        return self

    def transform(self, x):
        bs = x.shape[0]
        input_3d = x.ndim == 3

        if input_3d:
            # Shape (batch_size, n_samples, n_features)
            x = rearrange(x, "bs s f -> (bs s) f", bs=bs)

        # Calculate the quantile index for each sample and then linearly interpolate
        # between the two nearest quantiles

        # Find the last bucket where the quantile value is less than the value
        # We know that the quantiles are sorted in ascending order, therefore this is done by finding the number of quantiles with a value less than x
        # last_bucket_leq = (
        #     (self.quantiles[None, :-1, :] < x[:, None, :])
        #     .count_nonzero(dim=1)
        #     .clamp(1, self.n_quantiles - 1)
        # )
        # # count_nonzero is faster than summing a boolean tensor
        # # last_bucket_leq now has shape ((bs * s), f)
        # # We want to use this to index into the quantiles tensor

        # # We now know that x lies between the last_bucket_leq and last_bucket_leq + 1 quantiles
        # quantile_value = self.quantiles.gather(0, last_bucket_leq - 1)
        # next_quantile_value = self.quantiles.gather(0, last_bucket_leq)

        # # assert (x >= quantile_value) & (x <= next_quantile_value).all()

        # # Linearly interpolate between the two nearest quantiles
        # # Quantiles are non-decreasing, however two may be equal resulting in a division by zero
        # # If this is the case, we can set the interp_distance to 0.5 and wack the value in the middle of the two quantiles
        # # interp_distance = (x - quantile_value) / (next_quantile_value - quantile_value)

        # quantile_diffs = next_quantile_value - quantile_value

        # interp_distance = torch.where(
        #     quantile_diffs == 0,
        #     torch.tensor(0.5, device=x.device),
        #     (x - quantile_value) / quantile_diffs,
        # )

        # # Normalize this to a value between 0 and 1 to pass to the inverse CDF
        # x = ((last_bucket_leq + interp_distance) / self.n_quantiles).clamp(0, 1)

        x_values = self.quantiles

        last_x_value_lt_idx = (
            (x_values[None, :, :] < x[:, None, :])
            .sum(dim=1)
            .clamp(1, self.n_quantiles - 1)
        ) - 1

        # Count_nonzero sometimes causes OOM

        last_x_value_lt = x_values.gather(0, last_x_value_lt_idx)
        next_x_value = x_values.gather(0, last_x_value_lt_idx + 1)

        diff = next_x_value - last_x_value_lt
        interp_distance = torch.where(
            diff == 0, torch.tensor(0.5, device=x.device), (x - last_x_value_lt) / diff
        )

        y = ((last_x_value_lt_idx + interp_distance) / self.n_quantiles).clamp(0, 1)

        # Values that would be outside of the histogram are clamped to the ends because it was the easiest way to handle them

        if self.output_normal:
            y = self.norm_inv_cdf(y.clamp(0, 1)).clamp(-100, 100)

        if input_3d:
            y = rearrange(y, "(bs s) f -> bs s f", bs=bs)

        return y

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        input_3d = x.ndim == 3

        if input_3d:
            x = rearrange(x, "bs s f -> (bs s) f", bs=bs)

        if self.output_normal:
            x = self.norm_cdf(x)

        # probably not most efficient but makes gather work
        x_buckets = repeat(
            torch.linspace(0, 1, self.n_quantiles, device=x.device),
            "nb -> nb f",
            f=x.shape[-1],
        )

        last_x_value_lt_idx = (
            (x_buckets[None, :, :] < x[:, None, :])
            .sum(dim=1)
            .clamp(1, self.n_quantiles - 1)
        ) - 1

        last_x_bucket_lt = x_buckets.gather(0, last_x_value_lt_idx)
        next_x_bucket = x_buckets.gather(0, last_x_value_lt_idx + 1)

        diff = next_x_bucket - last_x_bucket_lt
        interp_distance = torch.where(
            diff == 0, torch.tensor(0.5, device=x.device), (x - last_x_bucket_lt) / diff
        )

        last_y_value_lt = self.quantiles.gather(0, last_x_value_lt_idx)
        next_y_value = self.quantiles.gather(0, last_x_value_lt_idx + 1)

        y = last_y_value_lt + interp_distance * (next_y_value - last_y_value_lt)

        if input_3d:
            y = rearrange(y, "(bs s) f -> bs s f", bs=bs)

        return y

    def norm_cdf(self, x):
        return 0.5 * (1 + torch.erf(x * 1 / math.sqrt(2)))

    def norm_inv_cdf(self, x):
        return torch.erfinv(2 * x - 1) * math.sqrt(2)
