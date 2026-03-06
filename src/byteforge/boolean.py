import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("boolean")
class Boolean(Encoding):
    """Single-bit boolean encoding.

    Encodes truthy/falsy values as 1/0.

    Args:
        bit_width: Must be 1.
        inverted: When True, truthy maps to 0 and falsy maps to 1.
    """

    def __init__(self, bit_width: int = 1, *, inverted: bool = False) -> None:
        if bit_width != 1:
            raise ValueError(f"Boolean bit_width must be 1, got {bit_width}")
        super().__init__(bit_width)
        self.inverted = inverted

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values).astype(bool).astype(np.uint8)
        if self.inverted:
            arr = np.uint8(1) - arr
        return arr

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        result = arr.astype(bool).astype(np.uint8)
        if self.inverted:
            result = np.uint8(1) - result
        return result

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, 1)

    def __repr__(self) -> str:
        return f"Boolean(inverted={self.inverted})"

    def __str__(self) -> str:
        return "Boolean(inverted)" if self.inverted else "Boolean"
