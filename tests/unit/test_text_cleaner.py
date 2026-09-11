from src.parsing.text_cleaner import TextCleaner


def test_clean_text_normalizes_bullets_and_whitespace() -> None:
    cleaner = TextCleaner()
    raw = "•  Led   a team of 12   engineers\n\n\n\n- Shipped v2 of the platform"
    cleaned = cleaner.clean_text(raw)
    assert "•" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Led a team of 12 engineers" in cleaned


def test_chunk_text_never_splits_a_line() -> None:
    cleaner = TextCleaner(max_chunk_chars=40, chunk_overlap_chars=0)
    text = "Led a team of 12 engineers\nShipped v2 of the platform\nReduced latency by 40 percent"
    chunks = cleaner.chunk_text(text)

    original_lines = set(text.split("\n"))
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert line == "" or line in original_lines


def test_chunk_text_handles_empty_input() -> None:
    cleaner = TextCleaner()
    assert cleaner.chunk_text("") == []
