"""Hypothesis property-based round-trip tests across all encoding types."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans, composite, floats, integers, sampled_from

from byteforge import (
    BCD,
    IEEE754,
    Boolean,
    DECFloat,
    DECFloatG,
    GrayCode,
    IBMFloat,
    LinearScaled,
    MilStd1750A,
    OffsetBinary,
    OnesComplement,
    TIFloat,
    TwosComplement,
    Unsigned,
)

_FUZZ_SETTINGS = settings(max_examples=200)

# Cap integer bit widths at 53 -- encode() converts via float64, losing
# precision beyond 2^53.
_MAX_INT_BW = 53


@composite
def unsigned_encoding_and_value(draw):
    bw = draw(integers(1, _MAX_INT_BW))
    enc = Unsigned(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def twos_complement_encoding_and_value(draw):
    bw = draw(integers(2, _MAX_INT_BW))
    enc = TwosComplement(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def ones_complement_encoding_and_value(draw):
    bw = draw(integers(2, _MAX_INT_BW))
    enc = OnesComplement(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def bcd_encoding_and_value(draw):
    bw = draw(sampled_from([w for w in range(4, 65, 4) if w <= _MAX_INT_BW]))
    enc = BCD(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def gray_code_encoding_and_value(draw):
    bw = draw(integers(1, _MAX_INT_BW))
    enc = GrayCode(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def offset_binary_encoding_and_value(draw):
    bw = draw(integers(1, _MAX_INT_BW))
    enc = OffsetBinary(bw)
    lo, hi = enc.value_range
    v = draw(integers(int(lo), int(hi)))
    return enc, v


@composite
def boolean_encoding_and_value(draw):
    inv = draw(booleans())
    enc = Boolean(inverted=inv)
    v = draw(sampled_from([0, 1]))
    return enc, v


_IEEE754_RANGES = {
    16: (-65504.0, 65504.0),
    32: (-3.4e38, 3.4e38),
    64: (-1.7e308, 1.7e308),
}


@composite
def ieee754_encoding_and_value(draw):
    bw = draw(sampled_from([16, 32, 64]))
    enc = IEEE754(bw)
    lo, hi = _IEEE754_RANGES[bw]
    v = draw(floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False))
    return enc, v


@composite
def linear_scaled_encoding_and_value(draw):
    bw = draw(integers(4, 32))
    signed = draw(booleans())
    scale = draw(floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    if draw(booleans()):
        scale = -scale
    offset = draw(
        floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    enc = LinearScaled(bw, scale_factor=scale, offset=offset, signed=signed)
    lo, hi = enc.value_range
    v = draw(floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False))
    return enc, v


_MIL1750_RANGES = {
    32: (-1e37, 1e37),
    48: (-1e37, 1e37),
}


@composite
def mil_std_1750a_encoding_and_value(draw):
    bw = draw(sampled_from([32, 48]))
    enc = MilStd1750A(bw)
    lo, hi = _MIL1750_RANGES[bw]
    v = draw(
        floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x) > 1e-30
        )
    )
    return enc, v


@composite
def ti_float_encoding_and_value(draw):
    bw = draw(sampled_from([32, 40]))
    enc = TIFloat(bw)
    v = draw(
        floats(min_value=-1e37, max_value=1e37, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x) > 1e-30
        )
    )
    return enc, v


@composite
def ibm_float_encoding_and_value(draw):
    bw = draw(sampled_from([32, 64]))
    enc = IBMFloat(bw)
    v = draw(
        floats(min_value=-1e37, max_value=1e37, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x) > 1e-30
        )
    )
    return enc, v


@composite
def dec_float_encoding_and_value(draw):
    bw = draw(sampled_from([32, 64]))
    enc = DECFloat(bw)
    v = draw(
        floats(min_value=-1e37, max_value=1e37, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x) > 1e-30
        )
    )
    return enc, v


@composite
def dec_float_g_encoding_and_value(draw):
    enc = DECFloatG()
    v = draw(
        floats(min_value=-1e37, max_value=1e37, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x) > 1e-30
        )
    )
    return enc, v


class TestRoundTripFuzz:
    """Property-based round-trip tests."""

    # Integer encodings: exact round-trip

    @_FUZZ_SETTINGS
    @given(unsigned_encoding_and_value())
    def test_unsigned(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(twos_complement_encoding_and_value())
    def test_twos_complement(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(ones_complement_encoding_and_value())
    def test_ones_complement(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(bcd_encoding_and_value())
    def test_bcd(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(gray_code_encoding_and_value())
    def test_gray_code(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(offset_binary_encoding_and_value())
    def test_offset_binary(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    @_FUZZ_SETTINGS
    @given(boolean_encoding_and_value())
    def test_boolean(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert int(result) == v

    # Float encodings: approximate round-trip

    _IEEE754_TINY = {16: 6.104e-5, 32: 1.175e-38, 64: 2.225e-308}

    @_FUZZ_SETTINGS
    @given(ieee754_encoding_and_value())
    def test_ieee754(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        if v == 0.0 or abs(v) < self._IEEE754_TINY[enc.bit_width]:
            assert abs(result - v) <= self._IEEE754_TINY[enc.bit_width]
        else:
            assert result == pytest.approx(v, rel=1e-2 if enc.bit_width == 16 else 1e-6)

    @_FUZZ_SETTINGS
    @given(linear_scaled_encoding_and_value())
    def test_linear_scaled(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        tol = 0.5 * abs(enc._scale_factor) + 1e-9
        assert result == pytest.approx(v, abs=tol)

    @_FUZZ_SETTINGS
    @given(mil_std_1750a_encoding_and_value())
    def test_mil_std_1750a(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        rel = 1e-5 if enc.bit_width == 32 else 1e-10
        assert result == pytest.approx(v, rel=rel)

    @_FUZZ_SETTINGS
    @given(ti_float_encoding_and_value())
    def test_ti_float(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        rel = 1e-5 if enc.bit_width == 32 else 1e-8
        assert result == pytest.approx(v, rel=rel)

    @_FUZZ_SETTINGS
    @given(ibm_float_encoding_and_value())
    def test_ibm_float(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        rel = 1e-5 if enc.bit_width == 32 else 1e-13
        assert result == pytest.approx(v, rel=rel)

    @_FUZZ_SETTINGS
    @given(dec_float_encoding_and_value())
    def test_dec_float(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        rel = 1e-5 if enc.bit_width == 32 else 1e-14
        assert result == pytest.approx(v, rel=rel)

    @_FUZZ_SETTINGS
    @given(dec_float_g_encoding_and_value())
    def test_dec_float_g(self, pair):
        enc, v = pair
        result = enc.decode(enc.encode(np.array([v])))[0]
        assert result == pytest.approx(v, rel=1e-14)

    # Cross-cutting: encode always in DN range

    @_FUZZ_SETTINGS
    @given(
        unsigned_encoding_and_value()
        | twos_complement_encoding_and_value()
        | ones_complement_encoding_and_value()
        | bcd_encoding_and_value()
        | gray_code_encoding_and_value()
        | offset_binary_encoding_and_value()
        | boolean_encoding_and_value()
    )
    def test_integer_encode_in_dn_range(self, pair):
        enc, v = pair
        dn = enc.encode(np.array([v]))[0]
        assert 0 <= int(dn) <= enc.max_unsigned

    @_FUZZ_SETTINGS
    @given(
        ieee754_encoding_and_value()
        | linear_scaled_encoding_and_value()
        | mil_std_1750a_encoding_and_value()
        | ti_float_encoding_and_value()
        | ibm_float_encoding_and_value()
        | dec_float_encoding_and_value()
        | dec_float_g_encoding_and_value()
    )
    def test_float_encode_in_dn_range(self, pair):
        enc, v = pair
        dn = enc.encode(np.array([v]))[0]
        assert 0 <= int(dn) <= enc.max_unsigned
