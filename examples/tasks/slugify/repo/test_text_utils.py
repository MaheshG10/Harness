from text_utils import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_and_spaces():
    assert slugify("  Python 3.12: What's New?  ") == "python-3-12-what-s-new"


def test_max_length_has_no_trailing_hyphen():
    assert slugify("aaaa bbbb cccc", max_length=10) == "aaaa-bbbb"


def test_only_symbols():
    assert slugify("!!!") == ""
