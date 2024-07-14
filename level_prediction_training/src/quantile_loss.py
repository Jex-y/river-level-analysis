import torch
from einops import repeat, reduce


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: torch.Tensor,
):
    errors = repeat(y_true, 'b l 1 -> b l q', q=len(quantiles)) - y_pred
    loss = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    return reduce(loss, 'b l q -> q', 'mean')
