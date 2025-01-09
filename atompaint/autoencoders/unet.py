from __future__ import annotations

import torch
import torch.nn as nn
import torchyield as ty

from atompaint.conditioning import forward_with_condition
from torch import Tensor
from torch.nn import Module
from escnn.nn import GeometricTensor, tensor_directsum

from typing import Iterable

class UNet(Module):

    def __init__(self, blocks: Iterable[UNetBlock]):
        super().__init__()
        for i, block in enumerate(blocks):
            self.add_module(str(i), block)

    def forward(self, x, y):
        """
        Arguments:
            x: The input image.
            y: The conditioning signal.

        Note that *x* may be a `Tensor` or a `GeometricTensor`.
        """
        skips = []

        for block in self.children():
            x = block(x, y, skips=skips)

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

    def forward(self, x, y, *, skips):
        raise NotImplementedError

    def _forward(self, x, y):
        # I want `UNetBlock` to accept layers that take both one input (x) and 
        # two (x, y).  That is, layers that ignore the conditioning signal and 
        # layers that don't.  This simplifies the API for end-users, since it 
        # allows them to mix-and-match layers regardless of their inputs.

        for wrappee in self.wrappees:
            x = forward_with_condition(wrappee, x, y)

        return x


class PushSkip(UNetBlock):

    def forward(self, x, y, *, skips):
        skip = self._forward(x, y)
        skips.append(skip)
        return skip

class PopAddSkip(UNetBlock):

    @staticmethod
    def adjust_in_channels(in_channels):
        return in_channels

    def forward(self, x, y, *, skips):
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

        return self._forward(x_skip, y)

class PopCatSkip(UNetBlock):

    @staticmethod
    def adjust_in_channels(in_channels):
        # Use the addition operator so that this method will also work when 
        # given a `escnn.nn.FieldType`.
        return in_channels + in_channels

    def forward(self, x, y, *, skips):
        x_orig = skips.pop()

        if _types(x_orig, x) == (GeometricTensor, GeometricTensor):
            x_skip = tensor_directsum([x_orig, x])

        elif _types(x_orig, x) == (GeometricTensor, Tensor):
            x_skip = torch.cat([x_orig.tensor, x], dim=1)

        elif _types(x_orig, x) == (Tensor, Tensor):
            x_skip = torch.cat([x_orig, x], dim=1)

        else:
            raise AssertionError

        return self._forward(x_skip, y)

class NoSkip(UNetBlock):

    def forward(self, x, y, *, skips):
        return self._forward(x, y)

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

