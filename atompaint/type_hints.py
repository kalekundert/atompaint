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
LayerFactory: TypeAlias = Callable[
        [FieldType, FieldType],
        Iterable[Module],
]

