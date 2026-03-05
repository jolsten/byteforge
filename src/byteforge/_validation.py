

def validate_bit_width(bit_width: int) -> None:
    """Validate that bit_width is an integer in [1, 64].

    Args:
        bit_width: The number of bits for the encoding.

    Raises:
        ValueError: If bit_width is not an integer in [1, 64].
    """
    if not isinstance(bit_width, int) or bit_width < 1 or bit_width > 64:
        raise ValueError(f"bit_width must be an integer in [1, 64], got {bit_width}")
