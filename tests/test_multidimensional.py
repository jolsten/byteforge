"""Test that all encodings preserve multi-dimensional array shapes."""

import numpy as np
import pytest

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

# Integer encodings with sample values within their range
_INTEGER_ENCODINGS = [
    (Unsigned(8), np.array([[0, 1, 2], [100, 200, 255]])),
    (TwosComplement(8), np.array([[-128, -1, 0], [1, 42, 127]])),
    (OnesComplement(8), np.array([[-127, -1, 0], [1, 42, 127]])),
    (OffsetBinary(8), np.array([[-128, -1, 0], [1, 42, 127]])),
    (GrayCode(8), np.array([[0, 1, 2], [100, 200, 255]])),
    (BCD(8), np.array([[0, 1, 9], [42, 50, 99]])),
    (Boolean(), np.array([[0, 1], [1, 0]])),
]

# Float encodings with sample values
_FLOAT_ENCODINGS = [
    (IEEE754(32), np.array([[0.0, 1.0, -1.0], [3.14, -100.5, 0.001]])),
    (
        LinearScaled(8, scale_factor=0.5, offset=0.0),
        np.array([[0.0, 1.0, 2.0], [50.0, 100.0, 127.0]]),
    ),
    (MilStd1750A(32), np.array([[0.0, 1.0, -1.0], [0.5, 100.25, -0.125]])),
    (TIFloat(32), np.array([[0.0, 1.0, -1.0], [3.14, -100.5, 0.5]])),
    (IBMFloat(32), np.array([[0.0, 1.0, -1.0], [0.1, -256.5, 42.0]])),
    (DECFloat(32), np.array([[0.0, 1.0, -1.0], [0.25, -50.0, 10.0]])),
    (DECFloatG(64), np.array([[0.0, 1.0, -1.0], [0.25, -50.0, 10.0]])),
]


class TestMultiDimensionalInteger:
    @pytest.mark.parametrize("enc,values", _INTEGER_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_encode_preserves_shape(self, enc, values):
        result = enc.encode(values)
        assert result.shape == values.shape

    @pytest.mark.parametrize("enc,values", _INTEGER_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_roundtrip_preserves_shape(self, enc, values):
        dns = enc.encode(values)
        result = enc.decode(dns)
        assert result.shape == values.shape

    @pytest.mark.parametrize("enc,values", _INTEGER_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_roundtrip_values(self, enc, values):
        dns = enc.encode(values)
        result = enc.decode(dns)
        np.testing.assert_array_equal(result, values)


class TestMultiDimensionalFloat:
    @pytest.mark.parametrize("enc,values", _FLOAT_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_encode_preserves_shape(self, enc, values):
        result = enc.encode(values)
        assert result.shape == values.shape

    @pytest.mark.parametrize("enc,values", _FLOAT_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_roundtrip_preserves_shape(self, enc, values):
        dns = enc.encode(values)
        result = enc.decode(dns)
        assert result.shape == values.shape

    @pytest.mark.parametrize("enc,values", _FLOAT_ENCODINGS, ids=lambda x: type(x).__name__ if hasattr(x, 'bit_width') else "")
    def test_roundtrip_approximate(self, enc, values):
        dns = enc.encode(values)
        result = enc.decode(dns)
        # Float encodings may lose precision; check non-zero values approximately
        nonzero = values != 0.0
        if np.any(nonzero):
            np.testing.assert_allclose(result[nonzero], values[nonzero], rtol=1e-2)


class TestHigherDimensional:
    """3D and higher shapes."""

    def test_3d_unsigned(self):
        values = np.arange(24).reshape(2, 3, 4)
        enc = Unsigned(8)
        dns = enc.encode(values)
        assert dns.shape == (2, 3, 4)
        result = enc.decode(dns)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(result, values)

    def test_3d_ieee754(self):
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        enc = IEEE754(64)
        dns = enc.encode(values)
        assert dns.shape == (2, 3, 4)
        result = enc.decode(dns)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(result, values)

    def test_4d_twos_complement(self):
        values = np.arange(-12, 12).reshape(2, 3, 2, 2)
        enc = TwosComplement(8)
        dns = enc.encode(values)
        assert dns.shape == (2, 3, 2, 2)
        result = enc.decode(dns)
        assert result.shape == (2, 3, 2, 2)
        np.testing.assert_array_equal(result, values)

    def test_to_bytes_multidim(self):
        values = np.array([[0, 255], [128, 1]])
        enc = Unsigned(8)
        dns = enc.encode(values)
        raw = enc.to_bytes(dns)
        assert raw.shape == (2, 2, 1)  # (*dns.shape, n_bytes)
        roundtrip = enc.decode(enc.from_bytes(raw))
        np.testing.assert_array_equal(roundtrip, values)

    def test_to_bytes_multidim_16bit(self):
        values = np.array([[256, 512], [1024, 0]])
        enc = Unsigned(16)
        dns = enc.encode(values)
        raw = enc.to_bytes(dns)
        assert raw.shape == (2, 2, 2)
        roundtrip = enc.decode(enc.from_bytes(raw))
        np.testing.assert_array_equal(roundtrip, values)
