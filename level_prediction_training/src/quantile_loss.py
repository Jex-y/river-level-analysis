import torch


def quantile_loss(
    y_true: torch.Tensor,
    y_pred_quantiles: torch.Tensor,
    quantiles: torch.Tensor,
):
    return torch.maximum(
        quantiles * (y_true - y_pred_quantiles),
        (quantiles - 1) * (y_true - y_pred_quantiles),
    ).mean()
