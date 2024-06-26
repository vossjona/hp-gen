import pytest
from preprocessing import _normalize_and_split_text


@pytest.mark.parametrize("input_text, expected_output", [
    ("Would've", ["would", "'ve"]),
    ("Hello, world!", ["hello", "world"]),
    ("is,was,are,were,'s,been,being,'re,'m,am,m", ["is", "was", "are", "were", "'s", "been", "being", "'re", "'m", "am", "m"]),
    ("aaaaargh argh", ["argh"])
    # Add more test cases here
])
def test_normalize_and_split_text(input_text, expected_output):
    assert _normalize_and_split_text(input_text) == expected_output


if __name__ == "__main__":
    pytest.main()