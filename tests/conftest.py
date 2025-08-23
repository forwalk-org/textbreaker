"""
Shared test configuration and fixtures.
"""
import pytest
from textbreaker import TextBreaker


@pytest.fixture
def default_breaker():
    """Standard TextBreaker for most tests."""
    return TextBreaker(line_length=50)


@pytest.fixture
def mobile_breaker():
    """Mobile-optimized TextBreaker."""
    return TextBreaker(line_length=25, balanced=1.5, tolerance=0.2)


@pytest.fixture
def flexible_breaker():
    """High-tolerance TextBreaker for edge cases."""
    return TextBreaker(line_length=30, tolerance=0.8)


@pytest.fixture
def sample_texts():
    """Common test texts."""
    return {
        'simple': "This is a simple test sentence.",
        'long': "This is a much longer sentence that should definitely need wrapping when processed by the text breaker.",
        'punctuation': "Hello, world! How are you? I'm fine, thanks.",
        'numbers': "The price is $123.45 and the quantity is 1,234 items.",
        'multilang': "This and that pero también y además mais aussi und oder.",
        'long_word': "The supercalifragilisticexpialidocious word is extraordinary."
    }
