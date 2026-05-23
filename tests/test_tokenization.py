from textbreaker.core import TextBreaker, TokenType


def test_token_splitting_punctuation_no_space():
    breaker = TextBreaker(line_length=10)
    # Testing punctuation followed by world without space
    text = "Hello.World"
    tokens = breaker._tokenize(text)

    # We want it to split 'Hello', '.', 'World'
    token_contents = [text[t.char_start:t.char_end] for t in tokens]
    assert "Hello" in token_contents
    assert "." in token_contents
    assert "World" in token_contents


def test_token_types_classification():
    breaker = TextBreaker(line_length=10)
    text = "Hello, World! and... more."
    tokens = breaker._tokenize(text)

    # Check if punctuation marks are correctly typed
    punctuation_tokens = [t for t in tokens if t.token_type == TokenType.PUNCTUATION]
    assert len(punctuation_tokens) >= 3

    # Check for breakword 'and'
    breakword_tokens = [t for t in tokens if t.token_type == TokenType.BREAKWORD]
    assert any(text[t.char_start:t.char_end].lower() == "and" for t in breakword_tokens)


def test_no_character_loss():
    breaker = TextBreaker(line_length=10)
    texts = [
        "Simple text.",
        "Text with... ellipsis and !? multiple marks.",
        "NoSpacesHere.AtAll",
        "  Leading and trailing whitespace  ",
        "\nNewlines\nincluded\n"
    ]

    for text in texts:
        tokens = breaker._tokenize(text)
        reconstructed = "".join(text[t.char_start:t.char_end] for t in tokens)
        assert reconstructed == text, f"Character loss for: '{text}'"


def test_numbers_behavior():
    # Just observing current behavior for numbers
    breaker = TextBreaker(line_length=10)
    text = "Pi is 3.14"
    tokens = breaker._tokenize(text)
    token_contents = [text[t.char_start:t.char_end] for t in tokens]
    # If it splits '3', '.', '14', that's what we expect after refine
    print(f"Number tokens: {token_contents}")
