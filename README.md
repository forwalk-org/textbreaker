# TextBreaker


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/forwalk-org/textbreaker/actions/workflows/ci.yml/badge.svg)](https://github.com/forwalk-org/textbreaker/actions/workflows/ci.yml)

An intelligent text breaking library that creates natural, readable line breaks by understanding linguistic patterns and respecting language boundaries. Unlike traditional text wrappers that break at arbitrary character positions, TextBreaker analyzes the structure of your text to find optimal break points based on punctuation, conjunctions, and word boundaries.

## Key Features

- **Linguistic Awareness**: Understands word types, punctuation, and natural break points
- **Multi-language Support**: Built-in support for English, Italian, Spanish, French, and German conjunctions
- **Balanced Line Lengths**: Avoids very short lines and creates visually pleasing text blocks
- **Configurable Tolerance**: Flexible line length constraints with customizable margins
- **Smart Tokenization**: Preserves text structure while enabling precise break point control

## Installation

TextBreaker is currently distributed directly through GitHub. To install it using pip, run:

```bash
pip install git+https://github.com/forwalk-org/textbreaker.git
```

## Quick Start

```python
from textbreaker import TextBreaker

# Basic usage with default settings
breaker = TextBreaker(line_length=50)
text = "This is a long sentence that needs to be wrapped intelligently at natural break points."
lines = breaker.wrap(text)

for line in lines:
    print(f"'{line}'")
# Output:
# 'This is a long sentence that needs to be'
# 'wrapped intelligently at natural break points.'
```

## Parameters

### `line_length: int`
**Target maximum characters per line**

The desired length for each line. Lines may slightly exceed this value within the tolerance bounds to find better break points.

```python
# Short lines for mobile interfaces
breaker = TextBreaker(line_length=30)

# Standard desktop width
breaker = TextBreaker(line_length=80)

# Wide format for print
breaker = TextBreaker(line_length=120)
```

### `balanced: float = 1.0`
**Line balancing strength (0.0 - 5.0)**

Controls how aggressively the wrapper tries to create evenly-sized lines and avoid very short final lines. Higher values produce more uniform line lengths.

- **0.0**: No balancing - prioritize breaking at best linguistic points
- **1.0**: Moderate balancing (default) - good balance between natural breaks and line uniformity  
- **2.0**: Strong balancing - creates more even line lengths
- **3.0+**: Maximum balancing - prioritizes uniform lines over natural break points

```python
# No balancing - pure linguistic breaks
breaker = TextBreaker(line_length=50, balanced=0.0)

# Light balancing
breaker = TextBreaker(line_length=50, balanced=0.5)

# Default moderate balancing
breaker = TextBreaker(line_length=50, balanced=1.0)

# Strong balancing for formal documents
breaker = TextBreaker(line_length=50, balanced=2.5)
```

### `tolerance: float = 0.3`
**Fractional flexibility around line_length (0.0 - 1.0)**

Defines how much shorter or longer lines can be relative to the target length when searching for optimal break points.

- **0.1**: Very strict - lines stay close to target length
- **0.3**: Default flexibility - good balance of consistency and natural breaks
- **0.5**: High flexibility - allows significant variation for better break points

```python
# Strict line lengths
breaker = TextBreaker(line_length=60, tolerance=0.1)

# Default flexibility  
breaker = TextBreaker(line_length=60, tolerance=0.3)

# High flexibility for natural breaks
breaker = TextBreaker(line_length=60, tolerance=0.6)
```

### `small_word_length: int = 3`
**Maximum length for small words (articles, prepositions)**

Words with length ≤ this value are classified as "small words" (articles, prepositions, etc.) which are generally less desirable break points.

```python
# Only single letters treated as small words
breaker = TextBreaker(line_length=50, small_word_length=1)

# Default - articles and short prepositions  
breaker = TextBreaker(line_length=50, small_word_length=3)

# Longer threshold for languages with longer articles
breaker = TextBreaker(line_length=50, small_word_length=4)
```

### `break_words: Optional[Set[str]] = None`
**Custom set of words indicating good break points**

Override or extend the default set of conjunctions and connectors that indicate natural pause points in text.

```python
# Use default multi-language break words
breaker = TextBreaker(line_length=50)

# Custom break words for technical text
tech_breaks = {'and', 'or', 'but', 'however', 'therefore', 'because'}
breaker = TextBreaker(line_length=50, break_words=tech_breaks)

# Extend default set
from textbreaker import TextBreaker
custom_breaks = TextBreaker.DEFAULT_BREAK_WORDS.copy()
custom_breaks.update(['however', 'therefore', 'moreover'])
breaker = TextBreaker(line_length=50, break_words=custom_breaks)
```

## Usage Examples

### Basic Text Wrapping

```python
breaker = TextBreaker(line_length=40)
text = "The quick brown fox jumps over the lazy dog and runs through the forest."
lines = breaker.wrap(text)

print('\n'.join(lines))
# Output:
# The quick brown fox jumps over the lazy
# dog and runs through the forest.
```

### Mobile-Friendly Formatting

```python
# Tight constraints for mobile screens
mobile_breaker = TextBreaker(
    line_length=25,
    balanced=1.5,
    tolerance=0.2
)

text = "Welcome to our mobile application! Please read the terms and conditions carefully."
lines = mobile_breaker.wrap(text)

for line in lines:
    print(f"|{line:<25}|")
# Output:
# |Welcome to our mobile    |
# |application! Please read |
# |the terms and conditions |
# |carefully.               |
```

### Academic Paper Formatting

```python
# Formal document with balanced paragraphs
academic_breaker = TextBreaker(
    line_length=72,
    balanced=2.0,
    tolerance=0.25
)

text = """The methodology employed in this research encompasses both qualitative and 
quantitative approaches to ensure comprehensive analysis of the data."""

lines = academic_breaker.wrap(text)
print('\n'.join(lines))
```

### Multi-language Support

```python
# Italian text
italian_text = "Questo è un esempio di testo italiano che deve essere formattato correttamente rispettando la struttura della lingua."
breaker = TextBreaker(line_length=50)
lines = breaker.wrap(italian_text)

# Spanish text  
spanish_text = "Este es un ejemplo de texto en español que necesita ser envuelto de manera inteligente."
lines = breaker.wrap(spanish_text)

# The breaker automatically recognizes conjunctions in multiple languages
```

### Custom Break Points for Code Documentation

```python
# Technical documentation with programming terms
code_breaks = {
    'and', 'or', 'but', 'when', 'then', 'else', 'if', 
    'while', 'for', 'function', 'method', 'class', 'object'
}

doc_breaker = TextBreaker(
    line_length=60,
    balanced=1.0,
    break_words=code_breaks
)

text = "This function takes a list of objects and processes each item using a custom method when the condition is met."
lines = doc_breaker.wrap(text)
print('\n'.join(lines))
```

## Default Break Words

The breaker includes built-in support for common conjunctions and connectors in multiple languages:

- **English**: and, but, or, so, yet, for, nor
- **Italian**: e, ma, o, però, quindi, né, oppure
- **Spanish**: y, pero, sino, ni, aunque, pues  
- **French**: et, mais, donc, car
- **German**: und, oder, aber, denn, doch, sondern, weder, noch

## Algorithm Overview

1. **Tokenization**: Text is split into classified tokens (words, punctuation, whitespace, etc.)
2. **Ranking**: Each token position receives a "break desirability" score based on linguistic patterns
3. **Optimization**: A windowing algorithm selects optimal break points considering both linguistic quality and line length constraints
4. **Assembly**: Final lines are constructed and cleaned up

## Requirements

- Python 3.7+
- No external dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/forwalk-org/textbreaker.git
cd textbreaker

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests locally in a Docker container using Tox (recommended)
./script/test-local.sh

# Run a specific Python environment (e.g. py310)
./script/test-local.sh -e py310

# Format code
black textbreaker/ tests/

# Type checking
mypy textbreaker/
```

### Running Tests

For local development, it is highly recommended to run tests inside the multi-python Docker container to ensure full compatibility across multiple Python environments without installing them locally:

```bash
# Run the entire test suite across all Python environments
./script/test-local.sh

# Run a specific Python environment (e.g., py310, py311)
./script/test-local.sh -e py310

# Pass additional arguments to pytest through tox (e.g., verbose mode or filter)
./script/test-local.sh -e py310 -- -v -k "test_tokenization"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Maurizio Melani** - [ForWalk Organization](https://github.com/forwalk-org)

## Changelog

### 1.0.0 (2024)
- Initial release
- Support for multiple languages
- Configurable line balancing
- Comprehensive test suite
- Full documentation

## Support

If you encounter any problems or have questions, please [open an issue](https://github.com/forwalk-org/textbreaker/issues) on GitHub.
