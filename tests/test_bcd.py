"""Tests for BCD encoding.

Reference vectors from type-convert/tests/test_bcd.py.
"""

import numpy as np
import pytest

from byteforge import BCD

# Reference decode vectors from type-convert (valid cases)
VALID_DECODE_VECTORS: list[tuple[int, int, int]] = [
    # (bit_width, dn, expected_value)
    (8, 0x03, 3),
    (8, 0x12, 12),
    (16, 0x1234, 1234),
    (16, 0x1986, 1986),
    (32, 0x12345678, 12345678),
    (32, 0x19860101, 19860101),
    (32, 0x20200501, 20200501),
    (64, 0x1234567890123456, 1234567890123456),
]

# Invalid BCD nibbles (should raise on decode)
INVALID_DECODE_VECTORS: list[tuple[int, int]] = [
    (8, 0x1A),
    (8, 0xA1),
    (8, 0xAA),
    (8, 0xFF),
    (16, 0x111A),
    (16, 0xA111),
    (16, 0xAAAA),
    (16, 0xFFFF),
    (32, 0x1111111A),
    (32, 0xA1111111),
    (32, 0xAAAAAAAA),
    (32, 0xFFFFFFFF),
    (64, 0x111111111111111A),
    (64, 0xA111111111111111),
    (64, 0xAAAAAAAAAAAAAAAA),
    (64, 0xFFFFFFFFFFFFFFFF),
]


@pytest.mark.parametrize("bit_width, dn, expected", VALID_DECODE_VECTORS)
def test_decode_reference_vectors(bit_width: int, dn: int, expected: int):
    enc = BCD(bit_width)
    result = enc.decode(np.array([dn], dtype=np.uint64))
    assert int(result[0]) == expected


@pytest.mark.parametrize("bit_width, dn, expected", VALID_DECODE_VECTORS)
def test_encode_roundtrip_reference(bit_width: int, dn: int, expected: int):
    """Encoding the decoded value should recover the original DN."""
    enc = BCD(bit_width)
    re_encoded = enc.encode(np.array([expected]))
    assert int(re_encoded[0]) == dn


@pytest.mark.parametrize("bit_width, dn", INVALID_DECODE_VECTORS)
def test_decode_invalid_nibble_raises(bit_width: int, dn: int):
    enc = BCD(bit_width)
    with pytest.raises(ValueError, match="Invalid BCD nibble"):
        enc.decode(np.array([dn], dtype=np.uint64))


class TestBCDEncode:
    def test_encode_single_digit(self):
        assert BCD(4).encode(np.array([7]))[0] == 7

    def test_encode_two_digits(self):
        assert BCD(8).encode(np.array([42]))[0] == 0x42

    def test_encode_three_digits(self):
        assert BCD(12).encode(np.array([123]))[0] == 0x123

    def test_encode_zero(self):
        assert BCD(8).encode(np.array([0]))[0] == 0x00

    def test_clamp_above_max(self):
        bcd = BCD(8)  # max is 99
        assert bcd.encode(np.array([200]))[0] == bcd.encode(np.array([99]))[0]

    def test_clamp_below_zero(self):
        assert BCD(8).encode(np.array([-5]))[0] == BCD(8).encode(np.array([0]))[0]

    def test_rounds_float(self):
        assert BCD(8).encode(np.array([42.7]))[0] == 0x43
        assert BCD(8).encode(np.array([42.2]))[0] == 0x42

    def test_encode_dtype(self):
        assert BCD(8).encode(np.array([42])).dtype == np.uint8
        assert BCD(16).encode(np.array([42])).dtype == np.uint16
        assert BCD(32).encode(np.array([42])).dtype == np.uint32
        assert BCD(64).encode(np.array([42])).dtype == np.uint64


class TestBCDDecode:
    def test_all_valid_8bit(self):
        """Every BCD-encoded value 0..99 should round-trip."""
        bcd = BCD(8)
        for v in range(100):
            dn = bcd.encode(np.array([v]))
            assert int(bcd.decode(dn)[0]) == v

    def test_4bit_all_values(self):
        bcd = BCD(4)
        for v in range(10):
            assert int(bcd.decode(bcd.encode(np.array([v])))[0]) == v

    def test_16bit_round_trip(self):
        bcd = BCD(16)
        for v in [0, 1234, 9999]:
            assert int(bcd.decode(bcd.encode(np.array([v])))[0]) == v


class TestBCDErrorsNaN:
    """decode_errors='nan' returns float64 with NaN for invalid elements."""

    def test_single_invalid_returns_nan(self):
        bcd = BCD(8, decode_errors="nan")
        result = bcd.decode(np.array([0xA5], dtype=np.uint64))
        assert result.dtype == np.float64
        assert np.isnan(result[0])

    def test_single_valid_returns_value(self):
        bcd = BCD(8, decode_errors="nan")
        result = bcd.decode(np.array([0x42], dtype=np.uint64))
        assert result.dtype == np.float64
        assert result[0] == 42.0

    def test_mixed_valid_invalid(self):
        bcd = BCD(8, decode_errors="nan")
        result = bcd.decode(np.array([0x12, 0xFF, 0x03, 0xAB], dtype=np.uint64))
        assert result.dtype == np.float64
        assert result[0] == 12.0
        assert np.isnan(result[1])
        assert result[2] == 3.0
        assert np.isnan(result[3])

    def test_all_valid_stays_float(self):
        bcd = BCD(8, decode_errors="nan")
        result = bcd.decode(np.array([0x00, 0x99], dtype=np.uint64))
        assert result.dtype == np.float64
        assert list(result) == [0.0, 99.0]

    @pytest.mark.parametrize("bit_width, dn", INVALID_DECODE_VECTORS)
    def test_all_invalid_vectors_return_nan(self, bit_width: int, dn: int):
        bcd = BCD(bit_width, decode_errors="nan")
        result = bcd.decode(np.array([dn], dtype=np.uint64))
        assert np.isnan(result[0])


class TestBCDErrorsSentinel:
    """decode_errors=<numeric> substitutes sentinel for invalid elements."""

    def test_single_invalid_returns_sentinel(self):
        bcd = BCD(8, decode_errors=255)
        result = bcd.decode(np.array([0xA5], dtype=np.uint64))
        assert result.dtype == np.uint8
        assert result[0] == 255

    def test_single_valid_returns_value(self):
        bcd = BCD(8, decode_errors=255)
        result = bcd.decode(np.array([0x42], dtype=np.uint64))
        assert result[0] == 42

    def test_mixed_valid_invalid(self):
        bcd = BCD(8, decode_errors=255)
        result = bcd.decode(np.array([0x12, 0xFF, 0x03, 0xAB], dtype=np.uint64))
        assert result.dtype == np.uint8
        assert list(result) == [12, 255, 3, 255]

    def test_zero_sentinel(self):
        bcd = BCD(8, decode_errors=0)
        result = bcd.decode(np.array([0xFF], dtype=np.uint64))
        assert result[0] == 0

    def test_16bit_sentinel(self):
        bcd = BCD(16, decode_errors=9999)
        result = bcd.decode(np.array([0xAAAA], dtype=np.uint64))
        assert result.dtype == np.uint16
        assert result[0] == 9999

    @pytest.mark.parametrize("bit_width, dn", INVALID_DECODE_VECTORS)
    def test_all_invalid_vectors_return_sentinel(self, bit_width: int, dn: int):
        sentinel = 200 if bit_width <= 8 else 12345
        bcd = BCD(bit_width, decode_errors=sentinel)
        result = bcd.decode(np.array([dn], dtype=np.uint64))
        assert result[0] == sentinel


class TestBCDErrorsValidation:
    """Valid inputs decode identically regardless of errors mode."""

    @pytest.mark.parametrize("bit_width, dn, expected", VALID_DECODE_VECTORS)
    def test_valid_same_as_raise(self, bit_width: int, dn: int, expected: int):
        sentinel = 200 if bit_width <= 8 else 9999
        for errors in ("raise", "nan", sentinel):
            result = BCD(bit_width, decode_errors=errors).decode(
                np.array([dn], dtype=np.uint64)
            )
            assert int(result[0]) == expected

    def test_invalid_errors_string(self):
        with pytest.raises(ValueError, match="decode_errors must be"):
            BCD(8, decode_errors="ignore")

    def test_sentinel_overflow_rejected(self):
        with pytest.raises(ValueError, match="does not fit"):
            BCD(8, decode_errors=9999)

    def test_negative_sentinel_rejected(self):
        with pytest.raises(ValueError, match="does not fit"):
            BCD(8, decode_errors=-1)

    def test_repr_with_errors(self):
        assert "decode_errors='nan'" in repr(BCD(8, decode_errors="nan"))
        assert "decode_errors=42" in repr(BCD(8, decode_errors=42))
        assert "decode_errors" not in repr(BCD(8))


class TestBCDMisc:
    def test_bit_width_must_be_multiple_of_4(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            BCD(5)

    def test_bit_width_out_of_range(self):
        with pytest.raises(ValueError, match="4-64"):
            BCD(0)
        with pytest.raises(ValueError, match="4-64"):
            BCD(68)

    def test_value_range(self):
        assert BCD(8).value_range == (0, 99)
        assert BCD(12).value_range == (0, 999)

    def test_repr(self):
        r = repr(BCD(8))
        assert "BCD" in r
        assert "8" in r
