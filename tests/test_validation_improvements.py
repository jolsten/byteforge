"""Tests for validation improvements and edge cases.

Covers:
- from_range() with invalid inputs (min >= max, equal values)
- 1-bit encodings (Unsigned(1), GrayCode(1), TwosComplement(1))
- encode_errors='nan' edge cases (all-NaN input, all out-of-range)
- encode_errors sentinel overflow validation
- dtype validation in decode (float arrays rejected)
"""

import numpy as np
import pytest

from byteforge import (
    BCD,
    Boolean,
    GrayCode,
    LinearScaled,
    OffsetBinary,
    OnesComplement,
    TwosComplement,
    Unsigned,
)

# -- from_range() invalid inputs -----------------------------------------------


class TestFromRangeValidation:
    """from_range() should reject min_value >= max_value."""

    def test_twos_complement_equal_values(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            TwosComplement.from_range(min_value=0, max_value=0)

    def test_twos_complement_reversed(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            TwosComplement.from_range(min_value=10, max_value=-10)

    def test_ones_complement_equal_values(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            OnesComplement.from_range(min_value=5, max_value=5)

    def test_ones_complement_reversed(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            OnesComplement.from_range(min_value=100, max_value=0)

    def test_offset_binary_equal_values(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            OffsetBinary.from_range(min_value=0, max_value=0)

    def test_offset_binary_reversed(self):
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            OffsetBinary.from_range(min_value=50, max_value=-50)

    def test_linear_scaled_equal_values(self):
        with pytest.raises(ValueError, match="physical_min must be less than physical_max"):
            LinearScaled.from_range(8, physical_min=5.0, physical_max=5.0)


# -- 1-bit encodings -----------------------------------------------------------


class TestOneBitEncodings:
    """Test 1-bit edge case for encodings beyond Boolean."""

    def test_unsigned_1bit_encode(self):
        enc = Unsigned(1)
        assert enc.encode(np.array([0]))[0] == 0
        assert enc.encode(np.array([1]))[0] == 1

    def test_unsigned_1bit_clamp(self):
        enc = Unsigned(1)
        assert enc.encode(np.array([5]))[0] == 1
        assert enc.encode(np.array([-1]))[0] == 0

    def test_unsigned_1bit_decode(self):
        enc = Unsigned(1)
        assert enc.decode(np.array([0], dtype=np.uint8))[0] == 0
        assert enc.decode(np.array([1], dtype=np.uint8))[0] == 1

    def test_unsigned_1bit_decode_out_of_range(self):
        enc = Unsigned(1)
        with pytest.raises(ValueError, match="out of range"):
            enc.decode(np.array([2], dtype=np.uint64))

    def test_unsigned_1bit_value_range(self):
        assert Unsigned(1).value_range == (0, 1)

    def test_unsigned_1bit_roundtrip(self):
        enc = Unsigned(1)
        for v in (0, 1):
            assert int(enc.decode(enc.encode(np.array([v])))[0]) == v

    def test_gray_code_1bit_encode(self):
        enc = GrayCode(1)
        assert enc.encode(np.array([0]))[0] == 0
        assert enc.encode(np.array([1]))[0] == 1

    def test_gray_code_1bit_decode(self):
        enc = GrayCode(1)
        assert enc.decode(np.array([0], dtype=np.uint8))[0] == 0
        assert enc.decode(np.array([1], dtype=np.uint8))[0] == 1

    def test_gray_code_1bit_clamp(self):
        enc = GrayCode(1)
        assert enc.encode(np.array([5]))[0] == 1
        assert enc.encode(np.array([-1]))[0] == 0

    def test_gray_code_1bit_value_range(self):
        assert GrayCode(1).value_range == (0, 1)

    def test_twos_complement_1bit(self):
        enc = TwosComplement(1)
        # 1-bit two's complement: only represents -1 and 0
        assert enc.value_range == (-1, 0)
        assert enc.encode(np.array([0]))[0] == 0
        assert enc.encode(np.array([-1]))[0] == 1
        assert enc.decode(np.array([0], dtype=np.uint8))[0] == 0
        assert enc.decode(np.array([1], dtype=np.uint8))[0] == -1

    def test_offset_binary_1bit(self):
        enc = OffsetBinary(1)
        assert enc.value_range == (-1, 0)
        assert enc.encode(np.array([0]))[0] == 1
        assert enc.encode(np.array([-1]))[0] == 0

    def test_ones_complement_2bit(self):
        # OnesComplement minimum is 2-bit (from from_range)
        enc = OnesComplement(2)
        assert enc.value_range == (-1, 1)


# -- encode_errors='nan' edge cases -------------------------------------------


class TestEncodeErrorsNanEdgeCases:
    """Edge cases for encode_errors='nan'."""

    def test_all_out_of_range(self):
        enc = Unsigned(8, encode_errors="nan")
        result = enc.encode(np.array([300, 400, -5]))
        assert result.dtype == np.float64
        assert all(np.isnan(result))

    def test_empty_array(self):
        enc = Unsigned(8, encode_errors="nan")
        result = enc.encode(np.array([], dtype=np.float64))
        assert len(result) == 0

    def test_single_out_of_range(self):
        enc = TwosComplement(8, encode_errors="nan")
        result = enc.encode(np.array([999]))
        assert result.dtype == np.float64
        assert np.isnan(result[0])

    def test_single_in_range(self):
        enc = TwosComplement(8, encode_errors="nan")
        result = enc.encode(np.array([50]))
        assert result.dtype == np.uint8
        assert result[0] == 50

    def test_nan_input_unsigned(self):
        enc = Unsigned(8, encode_errors="nan")
        result = enc.encode(np.array([np.nan]))
        # NaN clips to 0 via round+clip, but NaN < 0 and NaN > 255 are both False
        # so it won't be flagged as OOB — just the clamped result
        assert result.dtype == np.uint8

    def test_gray_code_nan(self):
        enc = GrayCode(8, encode_errors="nan")
        result = enc.encode(np.array([0, 256, 100]))
        assert result.dtype == np.float64
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_offset_binary_nan(self):
        enc = OffsetBinary(8, encode_errors="nan")
        result = enc.encode(np.array([0, 200, -200]))
        assert result.dtype == np.float64
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_ones_complement_nan(self):
        enc = OnesComplement(8, encode_errors="nan")
        result = enc.encode(np.array([0, 200, -200]))
        assert result.dtype == np.float64
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])


# -- encode_errors sentinel overflow -------------------------------------------


class TestEncodeErrorsSentinelValidation:
    """Numeric sentinel must fit in the output dtype."""

    def test_sentinel_too_large_for_uint8(self):
        enc = Unsigned(8, encode_errors=256)
        with pytest.raises(ValueError, match="does not fit"):
            enc.encode(np.array([300]))

    def test_sentinel_negative(self):
        enc = Unsigned(8, encode_errors=-1)
        with pytest.raises(ValueError, match="does not fit"):
            enc.encode(np.array([300]))

    def test_sentinel_fits_uint8(self):
        enc = Unsigned(8, encode_errors=255)
        result = enc.encode(np.array([300]))
        assert result[0] == 255

    def test_sentinel_fits_uint16(self):
        enc = Unsigned(16, encode_errors=65535)
        result = enc.encode(np.array([100000]))
        assert result[0] == 65535

    def test_sentinel_too_large_for_uint16(self):
        enc = Unsigned(16, encode_errors=65536)
        with pytest.raises(ValueError, match="does not fit"):
            enc.encode(np.array([100000]))

    def test_sentinel_zero_always_fits(self):
        enc = Unsigned(8, encode_errors=0)
        result = enc.encode(np.array([300]))
        assert result[0] == 0


# -- dtype validation in decode ------------------------------------------------


class TestDecodeDtypeValidation:
    """decode() should reject float arrays."""

    def test_unsigned_rejects_float(self):
        enc = Unsigned(8)
        with pytest.raises(TypeError, match="expects integer DNs"):
            enc.decode(np.array([1.5, 2.5]))

    def test_twos_complement_rejects_float(self):
        enc = TwosComplement(8)
        with pytest.raises(TypeError, match="expects integer DNs"):
            enc.decode(np.array([1.0]))

    def test_gray_code_rejects_float(self):
        enc = GrayCode(8)
        with pytest.raises(TypeError, match="expects integer DNs"):
            enc.decode(np.array([0.0], dtype=np.float32))

    def test_bcd_rejects_float(self):
        enc = BCD(8)
        with pytest.raises(TypeError, match="expects integer DNs"):
            enc.decode(np.array([0x42], dtype=np.float64))

    def test_boolean_rejects_float(self):
        enc = Boolean()
        with pytest.raises(TypeError, match="expects integer DNs"):
            enc.decode(np.array([0.5]))

    def test_int_array_still_works(self):
        enc = Unsigned(8)
        result = enc.decode(np.array([42], dtype=np.int32))
        assert result[0] == 42

    def test_uint_array_still_works(self):
        enc = Unsigned(8)
        result = enc.decode(np.array([42], dtype=np.uint8))
        assert result[0] == 42

    def test_python_int_scalar_still_works(self):
        enc = Unsigned(8)
        result = enc.decode(42)
        assert int(result) == 42
