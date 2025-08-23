"""
Test edge cases and special scenarios.
"""
import pytest
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
