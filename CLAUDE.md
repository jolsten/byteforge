# Byteforge

Encode and decode numbers as unsigned integer bit patterns (`np.ndarray`).

## Structure

- `src/byteforge/` — flat package
- `_base.py` — abstract `Encoding` base class (encode/decode always take/return `np.ndarray`)
- `_registry.py` — `@register` decorator and `create_encoding()` factory
- 13 encodings: Unsigned, TwosComplement, OnesComplement, IEEE754, LinearScaled, MilStd1750A, BCD, GrayCode, OffsetBinary, Boolean, TIFloat, IBMFloat, DECFloat/DECFloatG

## Commands

- **Test**: `uv run pytest tests/ -v`
- **Lint**: `uv run ruff check src/ tests/`
- **Type check**: `uv run mypy`

## C Extension (`src/byteforge/_c/`)

Optional NumPy C ufuncs accelerate encodings where Python has per-element loops or multi-pass numpy operations. Falls back to pure Python when the C extension is unavailable (set `BYTEFORGE_NO_C=1` to force fallback).

**Has C ufuncs:** BCD (encode/decode), GrayCode (decode only), MilStd1750A, TIFloat, IBMFloat, DECFloat/DECFloatG (all encode+decode).

**No C ufuncs:** Unsigned, TwosComplement, OnesComplement, IEEE754, OffsetBinary, Boolean, LinearScaled — these are already single-pass vectorized numpy ops (`view`, `clip`, `where`, arithmetic) with no loops or multi-pass overhead to eliminate. GrayCode encode is also pure Python since it's a single expression (`n ^ (n >> 1)`).

Each encoding module checks `_HAS_C` and dispatches to either the C ufunc or `_encode_py`/`_decode_py` helpers.

## Conventions

- Python >=3.9, numpy >=2.0
- Don't use from __future__ import annotations
- Google style docstrings; use type annotations, not types in docstrings
- Ruff: rules E, F, W, I, UP, B, SIM; line-length 100
- Tests use pytest + hypothesis
