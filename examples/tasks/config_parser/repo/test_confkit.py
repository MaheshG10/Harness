import pytest

from confkit import load_config


def test_simple():
    assert load_config("a = 1\nb=2") == {"a": "1", "b": "2"}


def test_comments_and_blank_lines():
    assert load_config("# top\n\nname = x  # trailing\n") == {"name": "x"}


def test_value_with_equals():
    assert load_config("url = https://x.io/?a=1&b=2") == {"url": "https://x.io/?a=1&b=2"}


def test_bad_line():
    with pytest.raises(ValueError):
        load_config("just text")
