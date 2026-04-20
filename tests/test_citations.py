from src.generation.citations import extract_cited_indices, validate_citations


def test_extract_indices():
    t = "First [1]. Second [2][3]."
    assert extract_cited_indices(t) == [1, 2, 3]


def test_validate():
    v, bad = validate_citations("See [1] and [99].", max_evidence=5)
    assert 1 in v
    assert 99 in bad
