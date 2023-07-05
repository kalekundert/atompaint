from typing import TypeAlias, Callable
from collections.abc import Iterable
from escnn.nn import FieldType, EquivariantModule

LayerFactory: TypeAlias = Callable[
        [FieldType, FieldType],
        Iterable[EquivariantModule],
]
