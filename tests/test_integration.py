"""
Integration tests for real-world scenarios.
"""
import pytest
from textbreaker import TextBreaker


class TestIntegration:
    """Test realistic usage scenarios."""

    def test_email_formatting(self):
        """Test email body formatting."""
        breaker = TextBreaker(line_length=72, balanced=1.5)
        text = ("Dear Customer, thank you for your order #12345. Your items will be "
                "shipped within 2-3 business days to the address provided.")

        result = breaker.wrap(text)
        assert len(result) >= 1
        assert all(len(line) <= 80 for line in result)

    def test_social_media_post(self):
        """Test social media post formatting."""
        breaker = TextBreaker(line_length=140, balanced=2.0)
        text = ("Just discovered this amazing new library for text formatting! "
                "It handles multiple languages and creates natural line breaks. "
                "#coding #python #textprocessing")

        result = breaker.wrap(text)
        assert len(result) >= 1

    def test_documentation_formatting(self):
        """Test technical documentation."""
        breaker = TextBreaker(line_length=80, balanced=1.0, tolerance=0.25)
        text = ("This function processes input data using advanced algorithms "
                "and returns formatted output that meets specified requirements.")

        result = breaker.wrap(text)
        assert len(result) >= 1

    def test_multilingual_document(self):
        """Test mixed-language document."""
        breaker = TextBreaker(line_length=60)
        text = ("Welcome to our service! Bienvenidos a nuestro servicio! "
                "Benvenuti al nostro servizio! Bienvenue à notre service!")

        result = breaker.wrap(text)
        assert len(result) >= 1

    def test_performance_large_text(self):
        """Test performance with larger text."""
        breaker = TextBreaker(line_length=50)
        large_text = "This is a test sentence. " * 50

        result = breaker.wrap(large_text)
        assert len(result) >= 5
        assert all(isinstance(line, str) for line in result)

    def test_configuration_comparison(self):
        """Test different configurations on same text."""
        text = ("This is a comprehensive test of the text breaking functionality "
                "with various configuration options to ensure optimal results.")

        configs = [
            TextBreaker(line_length=40, balanced=0.0),  # Pure linguistic
            TextBreaker(line_length=40, balanced=2.0),  # Balanced
            TextBreaker(line_length=40, tolerance=0.1), # Strict
            TextBreaker(line_length=40, tolerance=0.8)  # Flexible
        ]

        results = [breaker.wrap(text) for breaker in configs]

        # All should produce valid results
        assert all(len(result) >= 1 for result in results)
        assert all(isinstance(result, list) for result in results)
