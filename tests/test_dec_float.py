"""Tests for DECFloat and DECFloatG encodings.

Reference vectors from type-convert/tests/test_dec32.py, test_dec64.py, test_dec64g.py.
Reference: https://pubs.usgs.gov/of/2005/1424/
"""

import numpy as np
import pytest

from byteforge import DECFloat, DECFloatG

# ---------------------------------------------------------------------------
# DEC32 (F4) decode vectors
# ---------------------------------------------------------------------------
DECODE_VECTORS_32 = [
    (0x40800000, 1.0),
    (0xC0800000, -1.0),
    (0x41600000, 3.5),
    (0xC1600000, -3.5),
    (0x41490FD0, 3.141590),
    (0xC1490FD0, -3.141590),
    (0x7DF0BDC2, 9.9999999e36),
    (0xFDF0BDC2, -9.9999999e36),
    (0x03081CEA, 9.9999999e-38),
    (0x83081CEA, -9.9999999e-38),
    (0x409E0652, 1.234568),
    (0xC09E0652, -1.234568),
    (0x7FFFFFFF, 1.7014118e38),
    (0xFFFFFFFF, -1.7014118e38),
]

# ---------------------------------------------------------------------------
# DEC64 (D4) decode vectors
# ---------------------------------------------------------------------------
DECODE_VECTORS_64 = [
    (0x4080000000000000, 1.0),
    (0xC080000000000000, -1.0),
    (0x4160000000000000, 3.5),
    (0xC160000000000000, -3.5),
    (0x41490FDAA22168BE, 3.141592653589793),
    (0xC1490FDAA22168BE, -3.141592653589793),
    (0x7DF0BDC21ABB48DB, 1.0e37),
    (0xFDF0BDC21ABB48DB, -1.0e37),
    (0x03081CEA14545C75, 1.0e-37),
    (0x83081CEA14545C75, -1.0e-37),
    (0x409E06521462CEE7, 1.234567890123450),
    (0xC09E06521462CEE7, -1.234567890123450),
]

# ---------------------------------------------------------------------------
# DEC64G (G4) decode vectors
# ---------------------------------------------------------------------------
DECODE_VECTORS_64G = [
    (0x4010000000000000, 1.0),
    (0xC010000000000000, -1.0),
    (0x402C000000000000, 3.5),
    (0xC02C000000000000, -3.5),
    (0x402921FB54442D18, 3.141592653589793),
    (0xC02921FB54442D18, -3.141592653589793),
    (0x47BE17B84357691B, 1.0e37),
    (0xC7BE17B84357691B, -1.0e37),
    (0x3861039D428A8B8F, 1.0e-37),
    (0xB861039D428A8B8F, -1.0e-37),
    (0x4013C0CA428C59DD, 1.234567890123450),
    (0xC013C0CA428C59DD, -1.234567890123450),
]


class TestDECFloat32Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_32)
    def test_decode(self, dn, expected):
        enc = DECFloat(32)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestDECFloat64Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_64)
    def test_decode(self, dn, expected):
        enc = DECFloat(64)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestDECFloatGDecode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_64G)
    def test_decode(self, dn, expected):
        enc = DECFloatG()
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestDECFloatEncode:
    @pytest.mark.parametrize("bit_width", [32, 64])
    def test_encode_zero(self, bit_width):
        enc = DECFloat(bit_width)
        dn = enc.encode(np.array([0.0]))[0]
        assert int(dn) == 0

    @pytest.mark.parametrize("bit_width", [32, 64])
    @pytest.mark.parametrize("value", [1.0, -1.0, 3.5, -3.5, 0.5, 100.0])
    def test_encode_roundtrip(self, bit_width, value):
        enc = DECFloat(bit_width)
        dn = enc.encode(np.array([value]))
        result = enc.decode(dn)[0]
        rel = 1e-5 if bit_width == 32 else 1e-14
        assert result == pytest.approx(value, rel=rel)

    def test_encode_batch(self):
        enc = DECFloat(32)
        values = np.array([0.0, 1.0, -1.0, 3.5, 100.0])
        dns = enc.encode(values)
        decoded = enc.decode(dns)
        np.testing.assert_allclose(decoded, values, rtol=1e-5, atol=1e-38)


class TestDECFloatGEncode:
    def test_encode_zero(self):
        enc = DECFloatG()
        dn = enc.encode(np.array([0.0]))[0]
        assert int(dn) == 0

    @pytest.mark.parametrize("value", [1.0, -1.0, 3.5, -3.5, 0.5, 100.0])
    def test_encode_roundtrip(self, value):
        enc = DECFloatG()
        dn = enc.encode(np.array([value]))
        result = enc.decode(dn)[0]
        assert result == pytest.approx(value, rel=1e-14)


class TestDECFloatMisc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 32 or 64"):
            DECFloat(48)

    def test_dec_float_g_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 64"):
            DECFloatG(32)

    def test_repr_32(self):
        enc = DECFloat(32)
        assert "DECFloat" in repr(enc)
        assert "F4" in repr(enc)

    def test_repr_64(self):
        enc = DECFloat(64)
        assert "D4" in repr(enc)

    def test_repr_g(self):
        enc = DECFloatG()
        assert "DECFloatG" in repr(enc)
        assert "G4" in repr(enc)

    def test_dtype_32(self):
        enc = DECFloat(32)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint32

    def test_dtype_64(self):
        enc = DECFloat(64)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint64

    def test_dtype_g(self):
        enc = DECFloatG()
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint64

    @pytest.mark.parametrize("bit_width", [32, 64])
    def test_value_range(self, bit_width):
        enc = DECFloat(bit_width)
        lo, hi = enc.value_range
        assert lo < 0
        assert hi > 0
        assert lo == -hi

    def test_value_range_g(self):
        enc = DECFloatG()
        lo, hi = enc.value_range
        assert lo < 0
        assert hi > 0

    def test_decode_zero_exponent(self):
        """e=0 is reserved and should decode to zero."""
        enc = DECFloat(32)
        # s=0, e=0, m=nonzero — should still be zero
        dn = 0x00400000  # s=0, e=0, m=some nonzero bits
        result = enc.decode(np.array([dn]))[0]
        assert result == 0.0
