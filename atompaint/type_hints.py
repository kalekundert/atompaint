import torch

from typing import TypeAlias
from collections.abc import Iterable, Callable, Sequence

from torch.nn import Module
from escnn.nn import FieldType
from escnn.group import GroupElement

Grid: TypeAlias = Sequence[GroupElement]

ModuleFactory: TypeAlias = Callable[
        [FieldType],
        Module,
]
ConvFactory: TypeAlias = Callable[
        [FieldType, FieldType],
        Module,
]
PoolFactory: TypeAlias = Callable[
        [FieldType, int],
        Module,
]
LayerFactory: TypeAlias = Callable[
        [FieldType, FieldType],
        Iterable[Module],
]
OptFactory: TypeAlias = Callable[
        [Iterable[torch.nn.Parameter]],
        torch.optim.Optimizer,
]
LrFactory: TypeAlias = Callable[
        [torch.optim.Optimizer],
        torch.optim.lr_scheduler.LRScheduler,
]

