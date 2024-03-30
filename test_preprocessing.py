import pytest
from preprocessing import normalize_and_split_text


@pytest.mark.parametrize("input_text, expected_output", [
    ("Would've", ["would", "'ve"]),
    ("Hello, world!", ["hello", "world"]),
    ("is,was,are,were,'s,been,being,'re,'m,am,m", ["is", "was", "are", "were", "'s", "been", "being", "'re", "'m", "am", "m"]),
    # Add more test cases here
])
def test_normalize_and_split_text(input_text, expected_output):
    assert normalize_and_split_text(input_text) == expected_output


