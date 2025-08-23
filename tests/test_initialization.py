"""
Test TextBreaker initialization and parameter validation.
"""
import pytest
from textbreaker import TextBreaker


class TestInitialization:
    """Test parameter validation and setup."""

    def test_default_parameters(self):
        """Test default initialization."""
        breaker = TextBreaker(line_length=50)
        assert breaker.line_length == 50
        assert breaker.balanced == 1.0
        assert breaker.tolerance == 0.3

    def test_custom_parameters(self):
        """Test custom parameter setting."""
        breaker = TextBreaker(line_length=80, balanced=2.0, tolerance=0.5)
        assert breaker.line_length == 80
        assert breaker.balanced == 2.0
        assert breaker.tolerance == 0.5

    def test_invalid_line_length(self):
        """Test line_length validation."""
        with pytest.raises(ValueError):
            TextBreaker(line_length=0)

    def test_invalid_tolerance(self):
        """Test tolerance validation."""
        with pytest.raises(ValueError):
            TextBreaker(line_length=50, tolerance=1.5)

    def test_invalid_balanced(self):
        """Test balanced validation."""
        with pytest.raises(ValueError):
            TextBreaker(line_length=50, balanced=6.0)

    def test_boundary_values(self):
        """Test boundary parameter values."""
        breaker = TextBreaker(line_length=1, balanced=0.0, tolerance=0.0)
        assert breaker.line_length == 1

    def test_custom_break_words(self):
        """Test custom break words."""
        custom_breaks = {'custom', 'break'}
        breaker = TextBreaker(line_length=50, break_words=custom_breaks)
        assert breaker.break_words == custom_breaks

    def test_default_break_words(self):
        """Test default break words are loaded."""
        breaker = TextBreaker(line_length=50)
        assert 'and' in breaker.break_words
        assert 'but' in breaker.break_words
