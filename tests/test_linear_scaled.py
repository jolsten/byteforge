"""Tests for LinearScaled encoding."""

import numpy as np
import pytest

from byteforge import LinearScaled


class TestLinearScaledEncode:
    def test_unsigned_zero(self):
        enc = LinearScaled(16, scale_factor=360 / 65536)
        result = enc.encode(np.array([0.0]))
        assert result[0] == 0

    def test_unsigned_full_scale(self):
        enc = LinearScaled.from_range(16, physical_min=0.0, physical_max=360.0)
        result = enc.encode(np.array([360.0]))
        assert result[0] == 65535

    def test_unsigned_midpoint(self):
        enc = LinearScaled.from_range(16, physical_min=0.0, physical_max=360.0)
        dn = enc.encode(np.array([180.0]))[0]
        assert dn == pytest.approx(32768, abs=1)

    def test_clamp_above_max(self):
        enc = LinearScaled(8, scale_factor=1.0)
        result = enc.encode(np.array([1000.0]))
        assert result[0] == 255

    def test_clamp_below_min(self):
        enc = LinearScaled(8, scale_factor=1.0)
        result = enc.encode(np.array([-5.0]))
        assert result[0] == 0

    def test_with_offset(self):
        enc = LinearScaled(12, scale_factor=0.1, offset=-100.0)
        assert enc.encode(np.array([-100.0]))[0] == 0
        assert enc.encode(np.array([0.0]))[0] == 1000

    def test_signed_positive(self):
        enc = LinearScaled(8, scale_factor=1.0, signed=True)
        assert enc.encode(np.array([10]))[0] == 10

    def test_signed_negative(self):
        enc = LinearScaled(8, scale_factor=1.0, signed=True)
        assert enc.encode(np.array([-1]))[0] == 255

    def test_encode_dtype(self):
        assert LinearScaled(8, scale_factor=1.0).encode(np.array([0.0])).dtype == np.uint8
        assert LinearScaled(16, scale_factor=1.0).encode(np.array([0.0])).dtype == np.uint16
        assert LinearScaled(32, scale_factor=1.0).encode(np.array([0.0])).dtype == np.uint32
        assert LinearScaled(64, scale_factor=1.0).encode(np.array([0.0])).dtype == np.uint64


class TestLinearScaledDecode:
    def test_unsigned_roundtrip(self):
        enc = LinearScaled.from_range(16, physical_min=0.0, physical_max=360.0)
        for v in (0.0, 90.0, 180.0, 270.0):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v, abs=0.01)

    def test_signed_roundtrip(self):
        enc = LinearScaled(8, scale_factor=1.0, signed=True)
        for v in (-128, -1, 0, 1, 127):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v)

    def test_decode_with_offset(self):
        enc = LinearScaled(12, scale_factor=0.1, offset=-100.0)
        assert enc.decode(enc.encode(np.array([0.0])))[0] == pytest.approx(0.0, abs=0.05)
        assert enc.decode(enc.encode(np.array([-50.0])))[0] == pytest.approx(-50.0, abs=0.05)

    def test_decode_array(self):
        enc = LinearScaled.from_range(16, physical_min=0.0, physical_max=360.0)
        dns = np.array([0, 16384, 32768, 49151, 65535], dtype=np.uint64)
        arr = enc.decode(dns)
        for dn, dec in zip(dns, arr):
            assert dec == pytest.approx(
                enc.decode(np.array([int(dn)], dtype=np.uint64))[0], abs=1e-9
            )


class TestLinearScaledMisc:
    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            LinearScaled(8, scale_factor=0)

    def test_from_range_invalid_order_raises(self):
        with pytest.raises(ValueError, match="less than"):
            LinearScaled.from_range(8, physical_min=10.0, physical_max=5.0)

    def test_value_range(self):
        enc = LinearScaled.from_range(8, physical_min=0.0, physical_max=255.0)
        lo, hi = enc.value_range
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(255.0)

    def test_repr(self):
        r = repr(LinearScaled(8, scale_factor=1.0))
        assert "LinearScaled" in r
