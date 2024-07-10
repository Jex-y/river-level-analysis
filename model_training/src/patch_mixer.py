import torch
from torch import nn
import math
from .config import Config


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


# decomposition


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# pos_encoding


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(
    q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False
):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
        2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
        - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1), nn.GELU(), nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)  # x: [batch * n_val, a, d_model]
        return x


class PatchMixer(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.kernel_size = config.kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = (
            int((self.context_length - self.patch_len) / self.stride + 1) + 1
        )

        self.a = self.patch_num
        self.d_model = config.model_dim
        self.dropout = config.dropout
        self.head_dropout = config.head_dropout
        self.depth = config.encode_layers

        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(
                    dim=self.patch_num, a=self.a, kernel_size=self.kernel_size
                )
            )
        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.prediction_length),
            nn.Dropout(self.head_dropout),
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.prediction_length * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.prediction_length * 2), self.prediction_length),
            nn.Dropout(self.head_dropout),
        )
        self.dropout = nn.Dropout(self.dropout)

        self.reproject = nn.LazyLinear(1)

    def forward(self, x):
        batch_size = x.shape[0]
        num_features = x.shape[-1]

        x = x.permute(0, 2, 1)  # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)

        x = x_lookback.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # x: [batch, n_val, patch_num, patch_size]

        x = self.W_P(x)  # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)

        x = self.head1(x)
        x = u + x
        x = torch.reshape(
            x, (batch_size, num_features, -1)
        )  # x: [batch, n_val, pred_len]

        x = x.permute(0, 2, 1)

        x = self.reproject(x)

        return x
