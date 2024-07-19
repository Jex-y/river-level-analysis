import torch
from einops import repeat


def quantile_loss(
    y_true: torch.Tensor,
    y_pred_quantiles: torch.Tensor,
    quantiles: torch.Tensor,
):
    errors = repeat(y_true, "b l 1 -> b l q", q=len(quantiles)) - y_pred_quantiles
    loss = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    return torch.mean(loss) * 2
