import torch
import torchyield as ty

from torch import Tensor
from torch.nn import Module, ModuleList
from torchyield import LayerFactory

class AsymDenseBlock(Module):
    """
    A block that implements the core functionality of a DenseNet, i.e. 
    successively concatenating channels to the latent representation.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            concat_channels: int,
            concat_factories: list[LayerFactory],
            gather_factory: LayerFactory,
    ):
        """
        Arguments:
            concat_factories:
                - list of factories
                - arguments: input channels, channels to add
                - output type can be anything, will be concatenated to input 
                  tensor.
                - Typical: convolution, batch norm, activation

            gather_factory:
                - Produce modules that reduce number of channels and optionally 
                  reduce spatial dimensions.
                - Argument: input type and output type
                - Typical: 1x1x1 convolution to reduce channel dimension, then 
                  pool to reduce spatial dimensions.
        """
        super().__init__()

        curr_channels = in_channels
        concats = []

        for i, factory in enumerate(concat_factories):
            layer = factory(curr_channels, concat_channels)
            concat = ty.module_from_layers(layer)
            concats.append(concat)
            curr_channels += concat_channels

        self.concats = ModuleList(concats)
        self.gather = ty.module_from_layers(
                gather_factory(curr_channels, out_channels),
                verbose=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.concats:
            x = torch.cat([x, layer(x)], dim=1)
        return self.gather(x)


