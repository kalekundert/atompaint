from __future__ import annotations

import torch
import torch.nn as nn
import torchyield as ty
import inspect

from torch import Tensor
from torch.nn import Module
from escnn.nn import GeometricTensor, tensor_directsum

from typing import Iterable

class UNet(Module):

    def __init__(
            self,
            blocks: Iterable[UNetBlock],
            time_embedding: ty.Layer,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.time_embedding = ty.module_from_layers(time_embedding)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.time_embedding(t)

        skips = []

        for block in self.blocks:
            x = block(x, t, skips=skips)

        assert not skips

        return x

class UNetBlock(Module):
    """
    - Provide the means to connect the output of one module to the input of 
      another.
    - Account for the fact that not every module in the U-Net accepts the same 
      set of arguments (e.g. some are conditioned, others are not).
    """

    @classmethod
    def from_layers(cls, *layers):
        wrappees = ty.modules_from_layers(*layers)
        return cls(wrappees)

    def __init__(self, wrappees):
        super().__init__()
        self.wrappees = nn.ModuleList(wrappees)

    def forward(self, x, t, *, skips):
        raise NotImplementedError

    def _forward(self, x, t):
        # Some modules have only one input (x), while others have two (x, t).  
        # In the future, if I implement self-conditioning, this might get even 
        # more complicated.  I want to keep the simple API that allows users to 
        # yield modules with different inputs, but unfortunately I think that 
        # requires this somewhat messy code to inspect the module signatures in 
        # order to call them with the right arguments.

        for f in self.wrappees:
            # `torch.nn.Module.__call__()` accepts `*args` and `**kwargs`, so 
            # we need to inspect `forward()` instead.
            sig = inspect.signature(f.forward)

            # Some modules, for example `ConvTranspose3d`, accept an optional 
            # second argument.  We don't want to pass the `t` input to these 
            # modules.  This is why we first check to see if we can invoke the 
            # module with only one argument, and provide the second only if 
            # necessary.
            try:
                sig.bind(x)
            except TypeError:
                args = x, t
            else:
                args = x,

            x = f(*args)

        return x


class PushSkip(UNetBlock):

    def forward(self, x, t, *, skips):
        y = self._forward(x, t)
        skips.append(y)
        return y

class PopAddSkip(UNetBlock):

    @staticmethod
    def adjust_in_channels(in_channels):
        return in_channels

    def forward(self, x, t, *, skips):
        x_orig = skips.pop()

        # If `x` and `x_orig` are the both same type (either `Tensor` or 
        # `GeometricTensor`), then the addition operator will just work.  If 
        # not, then it must be that `x_orig` is the geometric tensor, because 
        # equivariance cannot be regained once lost.

        if _types(x_orig, x) == (GeometricTensor, GeometricTensor):
            x_skip = x_orig + x

        elif _types(x_orig, x) == (GeometricTensor, Tensor):
            x_skip = x_orig.tensor + x

        elif _types(x_orig, x) == (Tensor, Tensor):
            x_skip = x_orig + x

        else:
            raise AssertionError

        return self._forward(x_skip, t)

class PopCatSkip(UNetBlock):

    @staticmethod
    def adjust_in_channels(in_channels):
        # Use the addition operator so that this method will also work when 
        # given a `escnn.nn.FieldType`.
        return in_channels + in_channels

    def forward(self, x, t, *, skips):
        x_orig = skips.pop()

        if _types(x_orig, x) == (GeometricTensor, GeometricTensor):
            x_skip = tensor_directsum([x_orig, x])

        elif _types(x_orig, x) == (GeometricTensor, Tensor):
            x_skip = torch.cat([x_orig.tensor, x], dim=1)

        elif _types(x_orig, x) == (Tensor, Tensor):
            x_skip = torch.cat([x_orig, x], dim=1)

        else:
            raise AssertionError

        return self._forward(x_skip, t)

class NoSkip(UNetBlock):

    def forward(self, x, t, *, skips):
        return self._forward(x, t)

POP_SKIP_CLASSES = dict(
        cat=PopCatSkip,
        add=PopAddSkip,
)

def get_pop_skip_class(algorithm):
    try:
        return POP_SKIP_CLASSES[algorithm]
    except KeyError:
        raise ValueError(f"unknown skip algorithm: {algorithm}") from None

def _types(a, b):
    return type(a), type(b)

