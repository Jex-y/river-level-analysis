import equinox as eqx
from .config import Config

class MLPBlock(eqx.Module):
    def __init__(
        self,
        key,
        input_size,
        output_size,
        config: Config,
        is_last: bool = False,
    ):


        # super().__init__(
        #     nn.Linear(
        #         input_size,
        #         output_size,
        #         bias=not config.norm_before_activation and config.mlp_norm != Norm.NONE,
        #     ),
        #     conditional_layer(
        #         get_mlp_norm_layer(config, output_size),
        #         config.norm_before_activation,
        #     ),
        #     get_activation_function(config),
        #     conditional_layer(
        #         get_mlp_norm_layer(config, output_size),
        #         not config.norm_before_activation,
        #     ),
        #     get_dropout_layer(config) if not is_last else nn.Identity(),
        # )
