"""
Test advanced features and configuration options.
"""
import pytest
from textbreaker import TextBreaker


class TestAdvancedFeatures:
    """Test advanced configuration options."""

    def test_balanced_parameter(self, sample_texts):
        """Test balanced parameter effect."""
        text = sample_texts['long']

        breaker_low = TextBreaker(line_length=30, balanced=0.0)
        breaker_high = TextBreaker(line_length=30, balanced=3.0)

        result_low = breaker_low.wrap(text)
        result_high = breaker_high.wrap(text)

        assert len(result_low) >= 1
        assert len(result_high) >= 1

    def test_tolerance_parameter(self, sample_texts):
        """Test tolerance parameter effect."""
        text = sample_texts['long']

        breaker_strict = TextBreaker(line_length=25, tolerance=0.1)
        breaker_flexible = TextBreaker(line_length=25, tolerance=0.8)

        result_strict = breaker_strict.wrap(text)
        result_flexible = breaker_flexible.wrap(text)

        assert len(result_strict) >= 1
        assert len(result_flexible) >= 1

    def test_small_word_length(self):
        """Test small word length parameter."""
        text = "The quick brown fox jumps over the lazy dog"

        breaker_small = TextBreaker(line_length=20, small_word_length=1)
        breaker_large = TextBreaker(line_length=20, small_word_length=5)

        result_small = breaker_small.wrap(text)
        result_large = breaker_large.wrap(text)

        assert len(result_small) >= 1
        assert len(result_large) >= 1

    def test_mobile_optimization(self, sample_texts, mobile_breaker):
        """Test mobile-optimized settings."""
        result = mobile_breaker.wrap(sample_texts['long'])
        assert len(result) >= 2
        assert all(len(line) <= 35 for line in result)

    def test_multilanguage_support(self, default_breaker):
        """Test multi-language text."""
        texts = [
            "Questo è un esempio italiano che deve essere formattato",
            "Este es un ejemplo español que necesita formato",
            "Ceci est un exemple français qui nécessite formatage"
        ]

        for text in texts:
            result = default_breaker.wrap(text)
            assert len(result) >= 1

    def test_custom_break_words(self):
        """Test custom break words."""
        custom_breaks = {'custom', 'special', 'break'}
        breaker = TextBreaker(line_length=30, break_words=custom_breaks)

        text = "This custom break word should create breaks here"
        result = breaker.wrap(text)
        assert len(result) >= 1

    def test_extreme_line_lengths(self, sample_texts):
        """Test extreme line length values."""
        text = sample_texts['simple']

        # Very short
        breaker_short = TextBreaker(line_length=5)
        result_short = breaker_short.wrap(text)
        assert len(result_short) >= 1

        # Very long
        breaker_long = TextBreaker(line_length=200)
        result_long = breaker_long.wrap(text)
        assert len(result_long) == 1

    def test_zero_balanced_mode(self, sample_texts):
        """Test pure linguistic mode (balanced=0)."""
        breaker = TextBreaker(line_length=40, balanced=0.0)
        result = breaker.wrap(sample_texts['long'])
        assert len(result) >= 1
