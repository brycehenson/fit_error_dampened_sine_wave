def make_plotly_power_ten(
    power: int,
) -> str:
    """returns a power of ten in html format"""
    scaling_text = f"10<sup>{power:d}</sup>"
    return scaling_text
