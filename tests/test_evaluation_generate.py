import pytest

from evaluation.generate import (
    _distinctive_numbers,
    _pages_stating,
    _parse,
    sample_chunks,
)


@pytest.mark.parametrize(
    ("answer", "numbers"),
    [("34,367", ["34,367"]), ("$38.3 million", ["38.3"]), ("20", []), ("FY 2018", []),
     ("HSPD-12 PIV smart cards", []), ("1,242-page", ["1,242"]), ("2,500,000 titles in 2018", ["2,500,000"])],
)
def test_distinctive_numbers(answer, numbers):
    assert _distinctive_numbers(answer) == numbers


def test_pages_stating_only_uses_distinctive_numbers():
    pages = {1: "Titles cataloged 34,367 in 2018", 2: "in 2018 we had 20 staff", 3: "34,367 again"}
    assert _pages_stating("34,367", pages) == [1, 3]
    assert _pages_stating("20", pages) == []


def test_parse_handles_skip_and_garbage():
    assert _parse('{"question": "Q?", "answer": "A"}') == {"question": "Q?", "answer": "A"}
    assert _parse('{"skip": true}') is None
    assert _parse("not json") is None
    assert _parse('{"question": "Q?"}') is None


def test_sample_chunks_spreads_across_document_and_is_deterministic():
    chunks = [{"chunk_id": f"c{i:03d}", "text": "x" * 300, "page": i} for i in range(100)]
    chunks.append({"chunk_id": "short", "text": "too short", "page": 50})

    picked = sample_chunks(chunks, 10, seed=7)

    assert len(picked) == 10
    assert all(c["chunk_id"] != "short" for c in picked)
    assert [c["page"] // 10 for c in picked] == list(range(10))  # one per tenth of the document
    assert picked == sample_chunks(chunks, 10, seed=7)
