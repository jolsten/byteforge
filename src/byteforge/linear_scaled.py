from typing import Union

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("linear_scaled")
class LinearScaled(Encoding):
    """Encodes physical values as scaled integers using a linear transfer function.

    Transfer function::

        encode:  DN = clamp(round((EU - offset) / scale_factor), min_dn, max_dn)
        decode:  EU = scale_factor * DN + offset

    When ``signed=False`` (default), DNs range over ``[0, 2^N - 1]``.
    When ``signed=True``, DNs range over ``[-2^(N-1), 2^(N-1) - 1]``,
    but are stored as unsigned bit patterns using two's complement wrapping:
    negative DNs are mapped to ``[2^(N-1), 2^N - 1]`` on encode and
    unwrapped back on decode. This allows centering the physical range
    around a zero DN (e.g. a bipolar sensor with zero at mid-scale).

    Example (signed)::

        enc = LinearScaled(8, scale_factor=0.1, offset=0.0, signed=True)
        # DN range: -128..127, physical range: -12.8..12.7
        enc.encode([-1.0])  # -> DN -10, stored as unsigned 246

    Args:
        bit_width: Number of bits for the encoding.
        scale_factor: Scale factor for the transfer function (must be non-zero).
        offset: Offset for the transfer function.
        signed: If True, use signed DN range (two's complement wrapping).
    """

    def __init__(
        self,
        bit_width: int,
        *,
        scale_factor: float,
        offset: float = 0.0,
        signed: bool = False,
        encode_errors: Union[str, int, float] = "clamp",
    ) -> None:
        super().__init__(bit_width, encode_errors=encode_errors)
        if scale_factor == 0:
            raise ValueError("scale_factor must be non-zero")
        self._scale_factor = float(scale_factor)
        self._offset = float(offset)
        self._signed = signed
        if signed:
            self._min_dn = -(1 << (bit_width - 1))
            self._max_dn = (1 << (bit_width - 1)) - 1
        else:
            self._min_dn = 0
            self._max_dn = (1 << bit_width) - 1

    @classmethod
    def from_range(
        cls,
        bit_width: int,
        *,
        physical_min: float,
        physical_max: float,
        signed: bool = False,
    ) -> "LinearScaled":
        """Construct from a physical range, deriving scale_factor and offset.

        Args:
            bit_width: Number of bits for the encoding.
            physical_min: Minimum physical value.
            physical_max: Maximum physical value.
            signed: If True, use signed DN range.

        Returns:
            A LinearScaled encoding covering the given range.

        Raises:
            ValueError: If ``physical_min >= physical_max``.
        """
        if physical_min >= physical_max:
            raise ValueError("physical_min must be less than physical_max")
        n_steps = (1 << bit_width) - 1
        scale_factor = (physical_max - physical_min) / n_steps
        min_dn = -(1 << (bit_width - 1)) if signed else 0
        offset = physical_min - scale_factor * min_dn
        return cls(bit_width, scale_factor=scale_factor, offset=offset, signed=signed)

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = self.value_range
        dns = np.round((arr - self._offset) / self._scale_factor)
        dns = np.clip(dns, self._min_dn, self._max_dn).astype(np.int64)
        if self._signed:
            dns = np.where(dns < 0, dns + (1 << self.bit_width), dns)
        return self._apply_encode_overflow(arr, lo, hi, dns.astype(self._dn_dtype))

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        dns_i: np.ndarray = arr.astype(np.int64)
        if self._signed:
            dns_i = np.where(
                dns_i >= (1 << (self.bit_width - 1)),
                dns_i - (1 << self.bit_width),
                dns_i,
            )
        return (self._scale_factor * dns_i + self._offset).astype(np.float64)

    @property
    def value_range(self) -> tuple[float, float]:
        lo = self._scale_factor * self._min_dn + self._offset
        hi = self._scale_factor * self._max_dn + self._offset
        return (min(lo, hi), max(lo, hi))

    def __repr__(self) -> str:
        parts = [
            f"bit_width={self.bit_width}",
            f"scale_factor={self._scale_factor}",
            f"offset={self._offset}",
        ]
        if self._signed:
            parts.append("signed=True")
        return f"LinearScaled({', '.join(parts)})"

    def __str__(self) -> str:
        return f"LinearScaled({self.bit_width}, sf={self._scale_factor}, offset={self._offset})"
