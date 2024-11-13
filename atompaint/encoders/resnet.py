import torchyield as ty

from atompaint.utils import identity
from torch.nn import Module
from torchyield import Layer

class ResBlock(Module):

    def __init__(
            self,
            *,
            conv1,
            bn1,
            act1,
            conv2,
            bn2,
            act2,
            resize,
            resize_before_conv,
            skip,
            activation_before_skip,
    ):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.act1 = act1
        self.conv2 = conv2
        self.bn2 = bn2
        self.act2 = act2
        self.resize = resize
        self.resize_before_conv = resize_before_conv
        self.skip = skip
        self.activation_before_skip = activation_before_skip

    def forward(self, x):
        if self.resize_before_conv:
            x = self.resize(x)

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.activation_before_skip:
            y = self.act2(y)

        if not self.resize_before_conv:
            x = self.resize(x)

        z = self.skip(x) + y

        if not self.activation_before_skip:
            z = self.act2(z)

        return z

class SumLayer(Module):

    def __init__(self, left: Layer, right: Layer):
        super().__init__(self)
        self.left = ty.module_from_layer(left)
        self.right = ty.module_from_layer(right)

    def forward(self, x):
        return self.left(x) + self.right(x)

class SkipConnection(Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            body_factory,
            skip_factory,
    ):
        super().__init__(self)

        self.body = body_factory(in_channels, out_channels)

        if in_channels == out_channels:
            self.skip = identity
        else:
            self.skip = skip_factory(in_channels, out_channels)

    def forward(self, x):
        return self.body(x) + self.skip(x)

