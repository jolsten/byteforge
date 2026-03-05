

def validate_bit_width(bit_width: int) -> None:
    """Raise ValueError if *bit_width* is not in [1, 64]."""
    if not isinstance(bit_width, int) or bit_width < 1 or bit_width > 64:
        raise ValueError(f"bit_width must be an integer in [1, 64], got {bit_width}")
