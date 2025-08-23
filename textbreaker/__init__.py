"""
Natural Text Breaker - Intelligent text wrapping with natural language awareness.

This package provides advanced text wrapping capabilities that respect natural
language boundaries, support multiple languages, and create balanced,
readable line breaks.

Author: Maurizio Melani
License: MIT
Repository: https://github.com/forwalk-org/textbreaker
"""

from .core import TextBreaker, Token, TokenType

__version__ = "1.0.0"
__author__ = "Maurizio Melani"
__email__ = "maurizio@forwalk.org"
__license__ = "MIT"
__url__ = "https://github.com/forwalk-org/textbreaker"

__all__ = [
    "TextBreaker",
    "Token",
    "TokenType",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__"
]
