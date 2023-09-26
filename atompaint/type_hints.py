from typing import TypeAlias
from collections.abc import Iterable, Callable
from escnn.nn import FieldType, EquivariantModule

LayerFactory: TypeAlias = Callable[
        [FieldType, FieldType],
        Iterable[EquivariantModule],
]
