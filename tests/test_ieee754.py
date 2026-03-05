"""Tests for IEEE754 encoding."""

import struct

import numpy as np
import pytest

from byteforge import IEEE754


class TestIEEE754Encode:
    def test_32bit_one(self):
        enc = IEEE754(32)
        result = enc.encode(np.array([1.0]))
        expected = int.from_bytes(struct.pack(">f", 1.0), "big")
        assert result[0] == expected

    def test_32bit_zero(self):
        result = IEEE754(32).encode(np.array([0.0]))
        assert result[0] == 0

    def test_64bit(self):
        enc = IEEE754(64)
        result = enc.encode(np.array([3.14]))
        expected = int.from_bytes(struct.pack(">d", 3.14), "big")
        assert result[0] == expected

    def test_16bit(self):
        enc = IEEE754(16)
        result = enc.encode(np.array([1.0]))
        expected = int.from_bytes(struct.pack(">e", 1.0), "big")
        assert result[0] == expected

    def test_encode_dtype(self):
        assert IEEE754(16).encode(np.array([1.0])).dtype == np.uint16
        assert IEEE754(32).encode(np.array([1.0])).dtype == np.uint32
        assert IEEE754(64).encode(np.array([1.0])).dtype == np.uint64

    def test_batch(self):
        enc = IEEE754(32)
        vals = np.array([0.0, 1.0, -1.0])
        result = enc.encode(vals)
        assert len(result) == 3


class TestIEEE754Decode:
    def test_decode_32bit_one(self):
        enc = IEEE754(32)
        assert enc.decode(enc.encode(np.array([1.0])))[0] == pytest.approx(1.0)

    def test_decode_64bit(self):
        enc = IEEE754(64)
        assert enc.decode(enc.encode(np.array([3.14])))[0] == pytest.approx(3.14)

    def test_roundtrip_32(self):
        enc = IEEE754(32)
        for v in (0.0, -1.0, 3.14, 1e10):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v, rel=1e-6)

    def test_roundtrip_64(self):
        enc = IEEE754(64)
        for v in (0.0, -1.0, 3.14, 1e100):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v)

    def test_decode_array(self):
        enc = IEEE754(32)
        vals = np.array([0.0, 1.0, -1.0])
        encoded = enc.encode(vals)
        decoded = enc.decode(encoded)
        for orig, dec in zip(vals, decoded):
            assert dec == pytest.approx(orig, rel=1e-6)


class TestIEEE754SpecialValues:
    @pytest.mark.parametrize("bw", [16, 32, 64])
    def test_nan_roundtrip(self, bw):
        enc = IEEE754(bw)
        result = enc.decode(enc.encode(np.array([np.nan])))
        assert np.isnan(result[0])

    @pytest.mark.parametrize("bw", [16, 32, 64])
    def test_positive_inf_roundtrip(self, bw):
        enc = IEEE754(bw)
        result = enc.decode(enc.encode(np.array([np.inf])))[0]
        assert result == np.inf

    @pytest.mark.parametrize("bw", [16, 32, 64])
    def test_negative_inf_roundtrip(self, bw):
        enc = IEEE754(bw)
        result = enc.decode(enc.encode(np.array([-np.inf])))[0]
        assert result == -np.inf

    @pytest.mark.parametrize("bw", [32, 64])
    def test_negative_zero_roundtrip(self, bw):
        enc = IEEE754(bw)
        encoded = enc.encode(np.array([-0.0]))
        decoded = enc.decode(encoded)[0]
        assert decoded == 0.0
        assert np.signbit(decoded)

    def test_negative_zero_bit_pattern_32(self):
        enc = IEEE754(32)
        dn = enc.encode(np.array([-0.0]))[0]
        assert dn == 0x80000000

    def test_negative_zero_bit_pattern_64(self):
        enc = IEEE754(64)
        dn = enc.encode(np.array([-0.0]))[0]
        assert dn == 0x8000000000000000


class TestIEEE754ValueRange:
    @pytest.mark.parametrize("bw", [16, 32, 64])
    def test_value_range_finite(self, bw):
        enc = IEEE754(bw)
        lo, hi = enc.value_range
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    def test_value_range_16(self):
        lo, hi = IEEE754(16).value_range
        assert lo == pytest.approx(-65504.0)
        assert hi == pytest.approx(65504.0)

    def test_value_range_32(self):
        lo, hi = IEEE754(32).value_range
        assert lo == pytest.approx(-3.4028235e38, rel=1e-6)
        assert hi == pytest.approx(3.4028235e38, rel=1e-6)

    def test_value_range_64(self):
        lo, hi = IEEE754(64).value_range
        assert lo == pytest.approx(-1.7976931348623157e308, rel=1e-6)
        assert hi == pytest.approx(1.7976931348623157e308, rel=1e-6)

    @pytest.mark.parametrize("bw", [16, 32, 64])
    def test_value_range_symmetric(self, bw):
        lo, hi = IEEE754(bw).value_range
        assert lo == -hi


class TestIEEE754Misc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 16, 32, or 64"):
            IEEE754(8)

    def test_repr(self):
        r = repr(IEEE754(32))
        assert "IEEE754" in r
        assert "single" in r
