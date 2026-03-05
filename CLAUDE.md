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

## Conventions

- Python >=3.9, numpy >=1.26
- Don't use from __future__ import annotations
- Ruff: rules E, F, W, I, UP, B, SIM; line-length 100
- Tests use pytest + hypothesis
