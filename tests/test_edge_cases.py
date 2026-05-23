"""
Test edge cases and special scenarios.
"""
from textbreaker import TextBreaker


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_long_word(self, sample_texts, default_breaker):
        """Test single very long word."""
        result = default_breaker.wrap(sample_texts['long_word'])
        assert len(result) == 1
        assert "supercalifragilisticexpialidocious" in result[0]

    def test_minimum_length_constraint(self):
        """Test minimum length constraints."""
        breaker = TextBreaker(line_length=20)
        text = "a supercalifragilisticexpialidocious"
        result = breaker.wrap(text)

        assert len(result) >= 1
        full_result = ' '.join(result)
        assert "supercalifragilisticexpialidocious" in full_result

    def test_unicode_characters(self, default_breaker):
        """Test Unicode text handling."""
        text = "Café résumé naïve Zürich München"
        result = default_breaker.wrap(text)

        full_result = ' '.join(result)
        assert "Café" in full_result
        assert "München" in full_result

    def test_whitespace_only_text(self, default_breaker):
        """Test whitespace-only input."""
        result = default_breaker.wrap("   \n\t  ")
        assert result == ['']

    def test_consecutive_punctuation(self, default_breaker):
        """Test multiple consecutive punctuation marks."""
        text = "Really??? Yes!!! Absolutely!!!"
        result = default_breaker.wrap(text)

        full_result = ' '.join(result)
        assert "Really???" in full_result
        assert "Yes!!!" in full_result

    def test_mixed_number_formats(self, default_breaker):
        """Test various number formats."""
        text = "Prices: $1,234.56 €1.234,56 ¥12,345 coordinates 45.7749°N"
        result = default_breaker.wrap(text)

        full_result = ' '.join(result)
        assert "$1,234.56" in full_result or "$1" in full_result
        assert "45.7749" in full_result or "45" in full_result

    def test_urls_and_emails(self, default_breaker):
        """Test URLs and email addresses."""
        text = "Visit https://example.com or email test@example.com for info"
        result = default_breaker.wrap(text)

        full_result = ' '.join(result)
        assert "example.com" in full_result
        assert "test@example.com" in full_result or "test" in full_result

    def test_no_break_opportunities(self):
        """Test text with no good break opportunities."""
        breaker = TextBreaker(line_length=10)
        text = "supercalifragilisticexpialidocious"
        result = breaker.wrap(text)

        assert len(result) == 1
        assert result[0] == text

    def test_very_short_line_length(self):
        """Test with very small line length."""
        breaker = TextBreaker(line_length=1)
        text = "a b c d"
        result = breaker.wrap(text)

        # Should break at each space
        assert len(result) == 4
        assert result == ["a", "b", "c", "d"]

    def test_emoji_and_symbols(self):
        """Test text with emoji and symbols."""
        text = "Hello 🌍! This is a test 🚀 with many symbols 🛠️."
        breaker = TextBreaker(line_length=20)
        result = breaker.wrap(text)

        full_result = ' '.join(result)
        assert "🌍!" in full_result
        assert "🚀" in full_result

    def test_only_small_words(self):
        """Test text consisting only of small words."""
        text = "is in or by at"
        breaker = TextBreaker(line_length=5, small_word_length=3)
        result = breaker.wrap(text)

        # Even if they are all small, it should still wrap them
        assert len(result) > 1
        assert "".join(result).replace(" ", "") == text.replace(" ", "")

    def test_trailing_punctuation_only_line(self):
        """Test case where a line could end up with only punctuation."""
        text = "This is a sentence. !!!"
        breaker = TextBreaker(line_length=19)
        result = breaker.wrap(text)

        # 'This is a sentence.' is exactly 19 chars
        # '!!!' should be on the next line
        assert "!!!" in result[-1]
