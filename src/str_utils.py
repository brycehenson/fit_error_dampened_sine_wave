def make_plotly_power_ten(
    power: int,
) -> str:
    """returns a power of ten in html format"""
    str_power = f"{power:d}"
    minus_sign = "−"  # U+2212
    power_str = str_power.replace("-", minus_sign)
    scaling_text = f"10<sup>{power_str}</sup>"
    return scaling_text
