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
    and return ``np.ndarray``. Scalars are also accepted and returned
    as scalars.

    Args:
        bit_width: Number of bits for the encoding (1-64).
        encode_errors: Overflow behavior for ``encode()``:

            - ``"clamp"`` (default): silently clamp to the representable range
            - ``"raise"``: raise ``OverflowError`` if any value is out of range
            - ``"nan"``: return ``float64`` with ``np.nan`` for out-of-range elements
            - numeric value: substitute this sentinel for out-of-range elements

    Attributes:
        bit_width: Number of bits for the encoding.
        max_unsigned: Maximum unsigned value (``2^bit_width - 1``).
    """

    def __init__(
        self, bit_width: int, *, encode_errors: Union[str, int, float] = "clamp"
    ) -> None:
        if isinstance(encode_errors, str) and encode_errors not in ("clamp", "raise", "nan"):
            raise ValueError(
                f"encode_errors must be 'clamp', 'raise', 'nan', or a numeric value, "
                f"got {encode_errors!r}"
            )
        if not isinstance(encode_errors, (str, int, float)):
            raise TypeError(
                f"encode_errors must be str, int, or float, "
                f"got {type(encode_errors).__name__}"
            )
        validate_bit_width(bit_width)
        self.bit_width: int = bit_width
        self.max_unsigned: int = (1 << bit_width) - 1
        self._dn_dtype: type[np.unsignedinteger] = _min_uint_dtype(bit_width)
        self._encode_errors: Union[str, int, float] = encode_errors

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        """Encode physical values to unsigned integer bit patterns.

        Accepts scalars or arrays. If a scalar is passed, a scalar is returned.

        Args:
            values: Physical values to encode.

        Returns:
            Array with the minimum unsigned integer dtype that fits
            ``bit_width``, every element in ``[0, 2^bit_width - 1]``.
            Returns a scalar when given scalar input.
        """
        scalar = np.ndim(values) == 0
        result = self._encode(np.atleast_1d(np.asarray(values)))
        return result.item() if scalar else result

    @abstractmethod
    def _encode(self, values: np.ndarray) -> np.ndarray:
        """Encode physical values (array implementation).

        Subclasses implement this instead of ``encode()``.
        Input is guaranteed to be at least 1-D.
        """
        ...

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        """Decode unsigned integer bit patterns back to physical values.

        Accepts scalars or arrays. If a scalar is passed, a scalar is returned.

        Args:
            dns: Unsigned integer bit patterns to decode.

        Returns:
            Array of decoded physical values.
            Returns a scalar when given scalar input.

        Raises:
            ValueError: If any DN is outside ``[0, max_unsigned]``.
        """
        scalar = np.ndim(dns) == 0
        result = self._decode(np.atleast_1d(np.asarray(dns)))
        return result.item() if scalar else result

    @abstractmethod
    def _decode(self, dns: np.ndarray) -> np.ndarray:
        """Decode unsigned integer bit patterns (array implementation).

        Subclasses implement this instead of ``decode()``.
        Input is guaranteed to be at least 1-D.
        """
        ...

    def _validate_dns(self, dns: npt.ArrayLike) -> np.ndarray:
        """Validate DN values and return as uint64 array.

        Checks for negative values before casting to uint64 (avoiding silent
        wrapping), then checks the upper bound.

        Args:
            dns: DN values to validate.

        Returns:
            Validated array with dtype uint64.

        Raises:
            ValueError: If any DN is negative or exceeds ``max_unsigned``.
        """
        arr = np.asarray(dns)
        if np.isdtype(arr.dtype, "real floating"):
            raise TypeError(
                f"decode() expects integer DNs, got dtype {arr.dtype}. "
                f"Cast to an integer dtype first."
            )
        if np.isdtype(arr.dtype, "signed integer") and np.any(arr < 0):
            bad = arr[arr < 0]
            raise ValueError(
                f"DN value(s) {bad.ravel()[:5].tolist()} out of range for "
                f"{self!r} (expected 0-{self.max_unsigned})"
            )
        arr = arr.astype(np.uint64)
        if np.any(arr > self.max_unsigned):
            bad = arr[arr > self.max_unsigned]
            raise ValueError(
                f"DN value(s) {bad.ravel()[:5].tolist()} out of range for "
                f"{self!r} (expected 0-{self.max_unsigned})"
            )
        return arr

    def _apply_encode_overflow(
        self,
        values: np.ndarray,
        lo: Union[float, int],
        hi: Union[float, int],
        result: np.ndarray,
    ) -> np.ndarray:
        """Apply the encode overflow policy to the encoded result.

        Called after the encoding transform with both the original values
        (pre-clamp) and the encoded result (post-clamp). For ``"clamp"``
        mode the result is returned unchanged. Other modes substitute or
        raise based on out-of-bound elements.

        Args:
            values: Original input values (before clamping).
            lo: Lower bound of representable range (inclusive).
            hi: Upper bound of representable range (inclusive).
            result: The already-encoded DN array (post-clamp).

        Returns:
            The result array, potentially modified for nan/sentinel modes.

        Raises:
            OverflowError: If ``encode_errors='raise'`` and any value is
                out of range.
        """
        if self._encode_errors == "clamp":
            return result

        oob = (values < lo) | (values > hi)
        if not np.any(oob):
            return result

        if self._encode_errors == "raise":
            bad = np.asarray(values)[oob]
            raise OverflowError(
                f"Value(s) {bad.ravel()[:5].tolist()} outside representable range "
                f"[{lo}, {hi}] for {self!r}"
            )
        if self._encode_errors == "nan":
            out = result.astype(np.float64)
            out[oob] = np.nan
            return out
        # Numeric sentinel
        sentinel = self._encode_errors
        max_dn = np.iinfo(self._dn_dtype).max
        if int(sentinel) < 0 or int(sentinel) > max_dn:
            raise ValueError(
                f"Sentinel value {sentinel} does not fit in output dtype "
                f"{self._dn_dtype.__name__} (range 0-{max_dn})"
            )
        out = result.copy()
        out[oob] = self._dn_dtype(sentinel)
        return out

    def to_bytes(self, dns: npt.ArrayLike, byteorder: str = "big") -> np.ndarray:
        """Convert encoded DNs to raw bytes.

        For non-byte-aligned widths (e.g. 12-bit), values are zero-padded
        in the most significant bits to fill whole bytes.

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

        For non-byte-aligned widths, the most significant bits of the first
        byte (big-endian) or last byte (little-endian) are ignored.

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
            result |= b[..., i].astype(np.uint64) << np.uint64(8 * (n_bytes - 1 - i))
        return result.astype(self._dn_dtype)

    @property
    @abstractmethod
    def value_range(self) -> tuple[Union[float, int], Union[float, int]]:
        """Return the range of representable physical values.

        Returns:
            Tuple of ``(min_physical, max_physical)``.
        """
        ...

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return f"{type(self).__name__}(bit_width={self.bit_width})"

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.bit_width})"
