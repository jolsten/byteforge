"""Tests for new features: scalar handling, __eq__, __str__, overflow errors, from_range, registry."""

import numpy as np
import pytest

from byteforge import (
    BCD,
    IEEE754,
    Boolean,
    GrayCode,
    LinearScaled,
    OffsetBinary,
    OnesComplement,
    TwosComplement,
    Unsigned,
)
from byteforge._registry import register

# -- Scalar handling -----------------------------------------------------------


class TestScalarHandling:
    """encode(scalar) returns a scalar, encode(array) returns an array."""

    def test_unsigned_scalar_encode(self):
        result = Unsigned(8).encode(42)
        assert np.ndim(result) == 0
        assert int(result) == 42

    def test_unsigned_scalar_decode(self):
        result = Unsigned(8).decode(42)
        assert np.ndim(result) == 0
        assert int(result) == 42

    def test_twos_complement_scalar(self):
        enc = TwosComplement(8)
        dn = enc.encode(-1)
        assert np.ndim(dn) == 0
        val = enc.decode(dn)
        assert np.ndim(val) == 0
        assert int(val) == -1

    def test_ieee754_scalar(self):
        enc = IEEE754(32)
        dn = enc.encode(3.14)
        assert np.ndim(dn) == 0
        val = enc.decode(dn)
        assert np.ndim(val) == 0
        assert abs(float(val) - 3.14) < 1e-5

    def test_bcd_scalar(self):
        enc = BCD(8)
        dn = enc.encode(42)
        assert np.ndim(dn) == 0
        val = enc.decode(dn)
        assert np.ndim(val) == 0
        assert int(val) == 42

    def test_boolean_scalar(self):
        enc = Boolean()
        dn = enc.encode(True)
        assert np.ndim(dn) == 0
        assert int(dn) == 1

    def test_array_still_returns_array(self):
        result = Unsigned(8).encode(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


# -- __eq__ --------------------------------------------------------------------


class TestEquality:
    def test_same_encoding_equal(self):
        assert Unsigned(8) == Unsigned(8)

    def test_different_bit_width_not_equal(self):
        assert Unsigned(8) != Unsigned(16)

    def test_different_type_not_equal(self):
        assert Unsigned(8) != TwosComplement(8)

    def test_not_equal_to_non_encoding(self):
        assert Unsigned(8) != "Unsigned(8)"

    def test_twos_complement_equal(self):
        assert TwosComplement(16) == TwosComplement(16)

    def test_linear_scaled_equal(self):
        a = LinearScaled(8, scale_factor=0.5, offset=1.0)
        b = LinearScaled(8, scale_factor=0.5, offset=1.0)
        assert a == b

    def test_linear_scaled_different_sf(self):
        a = LinearScaled(8, scale_factor=0.5, offset=1.0)
        b = LinearScaled(8, scale_factor=1.0, offset=1.0)
        assert a != b

    def test_bcd_equal(self):
        assert BCD(8) == BCD(8)

    def test_bcd_different_errors(self):
        assert BCD(8) != BCD(8, decode_errors="nan")

    def test_boolean_equal(self):
        assert Boolean() == Boolean()

    def test_boolean_inverted_not_equal(self):
        assert Boolean() != Boolean(inverted=True)


# -- __str__ -------------------------------------------------------------------


class TestStr:
    def test_unsigned_str(self):
        assert str(Unsigned(8)) == "Unsigned(8)"

    def test_twos_complement_str(self):
        assert str(TwosComplement(16)) == "TwosComplement(16)"

    def test_boolean_str(self):
        assert str(Boolean()) == "Boolean"

    def test_boolean_inverted_str(self):
        assert str(Boolean(inverted=True)) == "Boolean(inverted)"

    def test_linear_scaled_str(self):
        s = str(LinearScaled(8, scale_factor=0.5, offset=1.0))
        assert "LinearScaled" in s
        assert "0.5" in s
        assert "1.0" in s

    def test_bcd_str_default(self):
        assert str(BCD(8)) == "BCD(8)"

    def test_bcd_str_with_errors(self):
        s = str(BCD(8, decode_errors="nan"))
        assert "nan" in s


# -- Overflow encode_errors="raise" --------------------------------------------


class TestOverflowRaise:
    def test_unsigned_raise(self):
        enc = Unsigned(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([256]))

    def test_unsigned_no_raise_in_range(self):
        enc = Unsigned(8, encode_errors="raise")
        result = enc.encode(np.array([255]))
        assert result[0] == 255

    def test_twos_complement_raise(self):
        enc = TwosComplement(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([128]))

    def test_twos_complement_raise_negative(self):
        enc = TwosComplement(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([-129]))

    def test_ones_complement_raise(self):
        enc = OnesComplement(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([128]))

    def test_offset_binary_raise(self):
        enc = OffsetBinary(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([128]))

    def test_gray_code_raise(self):
        enc = GrayCode(8, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([256]))

    def test_linear_scaled_raise(self):
        enc = LinearScaled(8, scale_factor=1.0, offset=0.0, encode_errors="raise")
        with pytest.raises(OverflowError):
            enc.encode(np.array([256.0]))

    def test_bcd_clamp_default(self):
        enc = BCD(8)
        result = enc.encode(np.array([100]))
        # 99 is max for 2-digit BCD
        assert int(enc.decode(result)[0]) == 99

    def test_invalid_errors_value(self):
        with pytest.raises(ValueError, match="encode_errors must be"):
            Unsigned(8, encode_errors="ignore")

    def test_clamp_default(self):
        # Default is clamp, should not raise
        enc = Unsigned(8)
        result = enc.encode(np.array([300]))
        assert result[0] == 255


# -- encode_errors="nan" and sentinel -----------------------------------------


class TestEncodeErrorsNaN:
    """encode_errors='nan' returns float64 with NaN for out-of-range elements."""

    def test_unsigned_nan(self):
        enc = Unsigned(8, encode_errors="nan")
        result = enc.encode(np.array([100, 300, 0, -5]))
        assert result.dtype == np.float64
        assert result[0] == 100
        assert np.isnan(result[1])
        assert result[2] == 0
        assert np.isnan(result[3])

    def test_twos_complement_nan(self):
        enc = TwosComplement(8, encode_errors="nan")
        result = enc.encode(np.array([0, 200, -1, -200]))
        assert result.dtype == np.float64
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])
        assert np.isnan(result[3])

    def test_all_in_range_keeps_dtype(self):
        enc = Unsigned(8, encode_errors="nan")
        result = enc.encode(np.array([0, 128, 255]))
        # All in range — no NaN needed, should still be uint8
        assert result.dtype == np.uint8


class TestEncodeErrorsSentinel:
    """encode_errors=<numeric> substitutes sentinel for out-of-range elements."""

    def test_unsigned_sentinel(self):
        enc = Unsigned(8, encode_errors=0)
        result = enc.encode(np.array([100, 300, 0]))
        assert result.dtype == np.uint8
        assert result[0] == 100
        assert result[1] == 0  # sentinel
        assert result[2] == 0

    def test_gray_code_sentinel(self):
        enc = GrayCode(8, encode_errors=255)
        result = enc.encode(np.array([10, 300]))
        assert result[1] == 255


# -- from_range() classmethods -------------------------------------------------


class TestFromRange:
    def test_twos_complement_from_range(self):
        enc = TwosComplement.from_range(min_value=-128, max_value=127)
        assert enc.bit_width == 8
        assert enc.value_range == (-128, 127)

    def test_twos_complement_from_range_16(self):
        enc = TwosComplement.from_range(min_value=-32768, max_value=32767)
        assert enc.bit_width == 16

    def test_ones_complement_from_range(self):
        enc = OnesComplement.from_range(min_value=-127, max_value=127)
        assert enc.bit_width == 8

    def test_offset_binary_from_range(self):
        enc = OffsetBinary.from_range(min_value=-128, max_value=127)
        assert enc.bit_width == 8

    def test_gray_code_from_range(self):
        enc = GrayCode.from_range(max_value=255)
        assert enc.bit_width == 8

    def test_gray_code_from_range_negative(self):
        with pytest.raises(ValueError, match="max_value must be >= 0"):
            GrayCode.from_range(max_value=-1)

    def test_bcd_from_range(self):
        enc = BCD.from_range(max_value=99)
        assert enc.bit_width == 8  # 2 digits * 4 bits

    def test_bcd_from_range_3_digits(self):
        enc = BCD.from_range(max_value=999)
        assert enc.bit_width == 12  # 3 digits * 4 bits

    def test_bcd_from_range_with_kwargs(self):
        enc = BCD.from_range(max_value=99, decode_errors="nan")
        assert enc._decode_errors == "nan"

    def test_bcd_from_range_negative(self):
        with pytest.raises(ValueError, match="max_value must be >= 0"):
            BCD.from_range(max_value=-1)


# -- Registry duplicate rejection ---------------------------------------------


class TestRegistryDuplicate:
    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match="already registered"):

            @register("unsigned")
            class DummyEncoding:
                pass
