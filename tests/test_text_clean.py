from src.ingestion.text_clean import normalize_whitespace, strip_repeated_headers_footers


def test_normalize_unicode_and_whitespace():
    s = "  Hello   world  \n\n\n  Next  "
    assert "  " not in normalize_whitespace(s)
    assert normalize_whitespace(s).startswith("Hello")


def test_strip_headers_single_page_noop():
    assert strip_repeated_headers_footers(["only one page"]) == ["only one page"]


def test_strip_headers_repeated_first_line():
    pages = [
        "HEADER\nbody one",
        "HEADER\nbody two",
        "HEADER\nbody three",
    ]
    out = strip_repeated_headers_footers(pages, min_runs=2)
    assert out[0].strip().startswith("body")
