from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt

from ._validation import validate_bit_width


def _min_uint_dtype(bits: int) -> type[np.unsignedinteger]:
    """Return the smallest unsigned numpy dtype that can hold the given bits.

    Args:
        bits: Number of bits to accommodate.

    Returns:
        The smallest unsigned integer dtype (uint8, uint16, uint32, or uint64).
    """
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    return np.uint64


def _min_int_dtype(bits: int) -> type[np.signedinteger]:
    """Return the smallest signed numpy dtype for the given two's complement width.

    Args:
        bits: Number of bits in the two's complement representation.

    Returns:
        The smallest signed integer dtype (int8, int16, int32, or int64).
    """
    if bits <= 8:
        return np.int8
    if bits <= 16:
        return np.int16
    if bits <= 32:
        return np.int32
    return np.int64


class Encoding(ABC):
    """Abstract base class for bit-width encoders.

    All encode/decode operations are fully vectorized -- they accept
    and return ``np.ndarray``. Scalar values should be wrapped in a
    1-element array by the caller.

    Args:
        bit_width: Number of bits for the encoding (1-64).

    Attributes:
        bit_width: Number of bits for the encoding.
        max_unsigned: Maximum unsigned value (``2^bit_width - 1``).
    """

    def __init__(self, bit_width: int) -> None:
        validate_bit_width(bit_width)
        self.bit_width: int = bit_width
        self.max_unsigned: int = (1 << bit_width) - 1
        self._dn_dtype: type[np.unsignedinteger] = _min_uint_dtype(bit_width)

    @abstractmethod
    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        """Encode physical values to unsigned integer bit patterns.

        Args:
            values: Physical values to encode.

        Returns:
            Array with the minimum unsigned integer dtype that fits
            ``bit_width``, every element in ``[0, 2^bit_width - 1]``.
        """
        ...

    @abstractmethod
    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        """Decode unsigned integer bit patterns back to physical values.

        Args:
            dns: Unsigned integer bit patterns to decode.

        Returns:
            Array of decoded physical values.

        Raises:
            ValueError: If any DN is outside ``[0, max_unsigned]``.
        """
        ...

    def _validate_dns(self, dns: np.ndarray) -> None:
        """Validate that all DN values are within range.

        Args:
            dns: Array of DN values to validate.

        Raises:
            ValueError: If any element exceeds ``max_unsigned``.
        """
        if np.any(dns > self.max_unsigned):
            bad = dns[dns > self.max_unsigned]
            raise ValueError(
                f"DN value(s) {bad[:5]} out of range for "
                f"{self.bit_width}-bit encoding (expected 0-{self.max_unsigned})"
            )

    def to_bytes(self, dns: npt.ArrayLike, byteorder: str = "big") -> np.ndarray:
        """Convert encoded DNs to raw bytes.

        Args:
            dns: Encoded bit patterns (output of ``encode()``).
            byteorder: ``"big"`` or ``"little"``.

        Returns:
            Array of dtype uint8 with shape ``(*dns.shape, n_bytes)``
            where ``n_bytes = ceil(bit_width / 8)``.
        """
        if byteorder not in ("big", "little"):
            raise ValueError(f"byteorder must be 'big' or 'little', got {byteorder!r}")
        n_bytes = (self.bit_width + 7) // 8
        arr = np.asarray(dns, dtype=np.uint64)
        cols = [
            ((arr >> (8 * i)) & 0xFF).astype(np.uint8)
            for i in reversed(range(n_bytes))
        ]
        result = np.stack(cols, axis=-1)
        if byteorder == "little":
            result = result[..., ::-1]
        return result

    def from_bytes(self, raw: npt.ArrayLike, byteorder: str = "big") -> np.ndarray:
        """Reconstruct encoded DNs from raw bytes.

        Args:
            raw: Byte array with shape ``(..., n_bytes)``, dtype uint8.
            byteorder: ``"big"`` or ``"little"``.

        Returns:
            Array with dtype matching ``_dn_dtype``.
        """
        if byteorder not in ("big", "little"):
            raise ValueError(f"byteorder must be 'big' or 'little', got {byteorder!r}")
        n_bytes = (self.bit_width + 7) // 8
        raw_arr = np.asarray(raw, dtype=np.uint8)
        if raw_arr.shape[-1] != n_bytes:
            raise ValueError(
                f"Expected {n_bytes} bytes per element for "
                f"{self.bit_width}-bit encoding, got {raw_arr.shape[-1]}"
            )
        b = raw_arr if byteorder == "big" else raw_arr[..., ::-1]
        result = np.zeros(raw_arr.shape[:-1], dtype=np.uint64)
        for i in range(n_bytes):
            result |= b[..., i].astype(np.uint64) << (8 * (n_bytes - 1 - i))
        return result.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[Union[float, int], Union[float, int]]:
        """Return the range of representable physical values.

        Returns:
            Tuple of ``(min_physical, max_physical)``.
        """
        lo: Union[float, int] = self.decode(np.array([0]))[0].item()
        hi: Union[float, int] = self.decode(np.array([self.max_unsigned]))[0].item()
        return (lo, hi)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(bit_width={self.bit_width})"
