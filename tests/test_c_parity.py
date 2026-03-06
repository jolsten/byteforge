"""Test that C-accelerated encode/decode matches pure-Python fallback."""

import os

import numpy as np
import pytest

from byteforge import BCD, DECFloat, DECFloatG, GrayCode, IBMFloat, MilStd1750A, TIFloat

# Skip the entire module if the C extension is not available.
pytestmark = pytest.mark.skipif(
    bool(os.environ.get("BYTEFORGE_NO_C")),
    reason="C extension disabled via BYTEFORGE_NO_C",
)


def _has_c(module) -> bool:
    return getattr(module, "_HAS_C", False)


# -- BCD ----------------------------------------------------------------------


class TestBCDParity:
    @pytest.fixture(params=[8, 16, 24])
    def enc(self, request):
        return BCD(request.param)

    def test_encode_parity(self, enc):
        if not _has_c(__import__("byteforge.bcd", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        max_val = min(enc._max_bcd_value, 9999)
        values = np.array([0, 1, 9, 42, max_val], dtype=np.uint64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.bcd", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        max_val = min(enc._max_bcd_value, 9999)
        values = np.array([0, 1, 9, 42, max_val], dtype=np.uint64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


# -- GrayCode (decode only) ---------------------------------------------------


class TestGrayCodeParity:
    @pytest.fixture(params=[4, 8, 16])
    def enc(self, request):
        return GrayCode(request.param)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.gray_code", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        dns = np.arange(min(enc.max_unsigned + 1, 256), dtype=np.uint64)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


# -- MilStd1750A --------------------------------------------------------------


class TestMilStd1750AParity:
    @pytest.fixture(params=[32, 48])
    def enc(self, request):
        return MilStd1750A(request.param)

    def test_encode_parity(self, enc):
        if not _has_c(__import__("byteforge.mil_std_1750a", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.5, 100.25, -0.125], dtype=np.float64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.mil_std_1750a", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.5, 100.25, -0.125], dtype=np.float64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


# -- TIFloat -------------------------------------------------------------------


class TestTIFloatParity:
    @pytest.fixture(params=[32, 40])
    def enc(self, request):
        return TIFloat(request.param)

    def test_encode_parity(self, enc):
        if not _has_c(__import__("byteforge.ti_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 3.14, -100.5], dtype=np.float64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.ti_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 3.14, -100.5], dtype=np.float64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


# -- IBMFloat ------------------------------------------------------------------


class TestIBMFloatParity:
    @pytest.fixture(params=[32, 64])
    def enc(self, request):
        return IBMFloat(request.param)

    def test_encode_parity(self, enc):
        if not _has_c(__import__("byteforge.ibm_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.1, -256.5], dtype=np.float64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.ibm_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.1, -256.5], dtype=np.float64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


# -- DECFloat / DECFloatG -----------------------------------------------------


class TestDECFloatParity:
    @pytest.fixture(params=[32, 64])
    def enc(self, request):
        return DECFloat(request.param)

    def test_encode_parity(self, enc):
        if not _has_c(__import__("byteforge.dec_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.25, -50.0], dtype=np.float64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self, enc):
        if not _has_c(__import__("byteforge.dec_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.25, -50.0], dtype=np.float64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)


class TestDECFloatGParity:
    def test_encode_parity(self):
        enc = DECFloatG(64)
        if not _has_c(__import__("byteforge.dec_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.25, -50.0], dtype=np.float64)
        py = enc._encode_py(values)
        c = enc._encode(values)
        np.testing.assert_array_equal(c, py)

    def test_decode_parity(self):
        enc = DECFloatG(64)
        if not _has_c(__import__("byteforge.dec_float", fromlist=["_HAS_C"])):
            pytest.skip("C extension not built")
        values = np.array([0.0, 1.0, -1.0, 0.25, -50.0], dtype=np.float64)
        dns = enc._encode_py(values)
        py = enc._decode_py(dns)
        c = enc._decode(dns)
        np.testing.assert_array_equal(c, py)
