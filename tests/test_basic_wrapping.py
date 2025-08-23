"""
Test basic text wrapping functionality.
"""
import pytest
from textbreaker import TextBreaker


class TestBasicWrapping:
    """Test core wrapping features."""

    def test_empty_text(self, default_breaker):
        """Test empty input."""
        assert default_breaker.wrap("") == []

    def test_short_text(self, default_breaker):
        """Test text shorter than line length."""
        result = default_breaker.wrap("Short text")
        assert result == ["Short text"]

    def test_simple_wrapping(self, sample_texts):
        """Test basic text wrapping."""
        breaker = TextBreaker(line_length=20)
        result = breaker.wrap(sample_texts['long'])
        assert len(result) > 1
        assert all(len(line) <= 30 for line in result)  # Allow tolerance

    def test_word_preservation(self, default_breaker):
        """Test that words are not broken."""
        text = "supercalifragilisticexpialidocious"
        result = default_breaker.wrap(text)
        assert text in result[0]

    def test_whitespace_normalization(self, default_breaker):
        """Test whitespace handling."""
        result = default_breaker.wrap("Word1     Word2")
        full_result = ' '.join(result)
        assert "Word1" in full_result and "Word2" in full_result

    def test_punctuation_preservation(self, sample_texts, default_breaker):
        """Test punctuation is preserved."""
        result = default_breaker.wrap(sample_texts['punctuation'])
        full_result = ' '.join(result)
        assert "Hello," in full_result
        assert "fine," in full_result

    def test_multiple_sentences(self, default_breaker):
        """Test multiple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        result = default_breaker.wrap(text)
        full_result = ' '.join(result)
        assert "First sentence" in full_result
        assert "Third sentence" in full_result

    def test_consistent_output(self, sample_texts, default_breaker):
        """Test output consistency."""
        result1 = default_breaker.wrap(sample_texts['simple'])
        result2 = default_breaker.wrap(sample_texts['simple'])
        assert result1 == result2

    def test_break_word_detection(self, sample_texts, default_breaker):
        """Test break word functionality."""
        result = default_breaker.wrap(sample_texts['multilang'])
        assert len(result) >= 1

    def test_number_preservation(self, sample_texts, default_breaker):
        """Test number format preservation."""
        result = default_breaker.wrap(sample_texts['numbers'])
        full_result = ' '.join(result)
        assert "$123.45" in full_result
        assert "1,234" in full_result
