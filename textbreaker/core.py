"""
Core module for Natural Text Breaker.

Contains the main TextBreaker class and supporting components for intelligent
text line breaking based on natural language patterns.
"""

import re
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Set


class TokenType(Enum):
    """
    Enumeration of token types found in natural language text.

    This classification helps the breaker make intelligent decisions about
    where to break lines by understanding the linguistic role of each token.

    Attributes:
        WORD: Regular words longer than the small_word_length threshold
        SMALLWORD: Short words like articles, prepositions (≤ small_word_length)
        PUNCTUATION: Sentence and clause punctuation marks
        WHITESPACE: Spaces, tabs, newlines, and other whitespace characters
        BREAKWORD: Special words that indicate good break points (conjunctions, connectors)
        OTHER: Numbers, symbols, and unclassified characters
    """
    WORD = auto()
    SMALLWORD = auto()
    PUNCTUATION = auto()
    WHITESPACE = auto()
    BREAKWORD = auto()
    OTHER = auto()


@dataclass
class Token:
    """
    Represents a single token in the text with metadata for intelligent wrapping.

    A token is the smallest unit of text processing, containing information
    about its position, type, characteristics, and break desirability that
    influence line breaking decisions.

    Attributes:
        seq (int): Sequential token number in the text
        token_type (TokenType): Classification of the token's linguistic role
        length (int): Number of characters in the token
        char_start (int): Starting character position in the original text
        separation_rank (int): Break desirability rank for this token position
    """
    seq: int
    token_type: TokenType
    length: int
    char_start: int
    separation_rank: int = field(default=0, init=False)

    @property
    def char_end(self) -> int:
        """Get the ending character position (exclusive) of this token."""
        return self.char_start + self.length


class TextBreaker:
    """
    Advanced text breaker that creates natural, readable line breaks.

    This breaker analyzes text structure to identify optimal break points based on
    linguistic patterns, token types, and configurable parameters. It supports
    multiple languages and can balance line lengths for improved readability.

    The algorithm works in several phases:
    1. Tokenization: Split text into classified tokens with character positions
    2. Ranking: Assign break desirability scores to each token
    3. Optimization: Select break points using token-based windowing algorithm
    4. Assembly: Construct final wrapped lines

    Args:
        line_length (int): Target maximum characters per line
        balanced (float): Line balancing strength (0.0-5.0). Higher values create more
                         evenly sized lines and avoid very short final lines.
                         0.0 = no balancing, 1.0 = moderate balancing, 3.0+ = strong balancing
        tolerance (float): Fractional slack around line_length (0.0-1.0)
        small_word_length (int): Maximum length to classify words as SMALLWORD
        break_words (set): Custom set of words that indicate good break points

    Example:
        >>> breaker = TextBreaker(line_length=50, balanced=1.5)
        >>> text = "This is a long sentence that needs intelligent wrapping."
        >>> lines = breaker.wrap(text)
        >>> for line in lines:
        ...     print(f"'{line}'")
        'This is a long sentence that needs'
        'intelligent wrapping.'
    """

    # Default break words (conjunctions and connectors) for multiple languages
    # These words typically indicate good break points when preceded by whitespace
    DEFAULT_BREAK_WORDS = {
        # English coordinating conjunctions
        'and', 'but', 'or', 'so', 'yet', 'for', 'nor',

        # Italian conjunctions
        'e', 'ma', 'o', 'però', 'quindi', 'né', 'oppure',

        # Spanish conjunctions
        'y', 'pero', 'sino', 'ni', 'aunque', 'pues',

        # French conjunctions
        'et', 'mais', 'donc', 'car',

        # German conjunctions
        'und', 'oder', 'aber', 'denn', 'doch', 'sondern',
        'weder', 'noch'
    }

    def __init__(self,
                 line_length: int,
                 balanced: float = 1.0,
                 tolerance: float = 0.3,
                 small_word_length: int = 3,
                 break_words: Optional[Set[str]] = None) -> None:
        """
        Initialize the TextBreaker with configuration parameters.

        Args:
            line_length: Target maximum characters per line. Lines may exceed
                        this slightly within tolerance bounds.
            balanced: Line balancing strength as a float value (0.0-5.0). Higher values
                     apply stronger distance penalties to create more evenly sized lines
                     and avoid very short final lines. 0.0 disables balancing completely.
            tolerance: Fractional flexibility around line_length. For example,
                      0.3 allows lines 30% shorter or longer than target.
            small_word_length: Words with length ≤ this value are classified
                              as SMALLWORD (typically articles, prepositions).
            break_words: Set of words that should be classified as BREAKWORD tokens,
                        indicating good break points. If None, uses DEFAULT_BREAK_WORDS
                        containing common conjunctions and connectors.

        Raises:
            ValueError: If line_length <= 0, tolerance < 0 or tolerance > 1,
                       or balanced < 0 or balanced > 5
        """
        if line_length <= 0:
            raise ValueError("line_length must be positive")
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("tolerance must be between 0.0 and 1.0")
        if not 0.0 <= balanced <= 5.0:
            raise ValueError("balanced must be between 0.0 and 5.0")
        if small_word_length < 0:
            raise ValueError("small_word_length must be non-negative")

        self.line_length = line_length
        self.balanced = balanced
        self.tolerance = tolerance
        self.small_word_length = small_word_length
        self.break_words = break_words if break_words is not None else self.DEFAULT_BREAK_WORDS.copy()

    def wrap(self, text: str) -> List[str]:
        """
        Break input text into optimally wrapped lines.

        This is the main entry point for text wrapping. The method analyzes
        the text structure, identifies break points, and returns a list of
        lines that respect natural language boundaries while meeting length
        constraints.

        Args:
            text: The raw text string to wrap. May contain any Unicode characters.

        Returns:
            A list of strings representing the wrapped lines. Each line is
            stripped of leading and trailing whitespace. Empty lines are
            filtered out unless they were intentionally present in the input.
        """
        if not text:
            return []

        if not text.strip():
            return ['']

        # Tokenize text into classified units with positions
        tokens = self._tokenize(text)

        # Calculate separation ranks for each token
        self._calculate_separation_ranks(tokens)

        # Find optimal break points using efficient token algorithm
        breaks = self._find_break_points(tokens)

        # Handle edge case: no breaks or single break at position 0
        if not breaks or len(breaks) == 1:
            return [text.strip()]

        # Slice text at break positions and clean up lines
        lines, start = [], 0
        for bp in breaks[1:]:  # Skip first break (position 0)
            line = text[start:bp].strip()
            if line:  # Only add non-empty lines
                lines.append(line)
            start = bp

        # Add final segment if it exists
        if start < len(text):
            final_line = text[start:].strip()
            if final_line:
                lines.append(final_line)

        return lines

    def _tokenize(self, text: str) -> List[Token]:
        """
        Split text into classified tokens for intelligent break point analysis.

        This method uses regular expressions to identify different types of
        text elements and classify them according to their linguistic role.
        Each token tracks its position in the original text for precise breaking.

        Args:
            text: Raw string to tokenize

        Returns:
            Ordered list of Token objects representing the text structure with positions
        """
        # Pattern to recognize numbers with commas and dots as separators
        # Captures numbers like: 1.045, 1,567, 1.1.1, 2.3, 123.456.789, etc.
        number_pattern = r'\d+(?:[.,]\d+)*'

        # Complete pattern: numbers with separators, words, whitespace, or other characters
        # Order is important: numbers first, then the rest
        pattern = f'({number_pattern}|\\w+|\\s+|[^\\w\\s]+)'
        raw_tokens = re.findall(pattern, text)
        tokens: List[Token] = []
        char_pos = 0

        for tid, part in enumerate(raw_tokens):
            low = part.lower()

            # Token classification based on characteristics
            if part.isspace():
                ttype = TokenType.WHITESPACE
            elif re.fullmatch(number_pattern, part):
                # Numbers with commas and dots are classified as OTHER
                ttype = TokenType.OTHER
            elif low in self.break_words:
                ttype = TokenType.BREAKWORD  # Conjunctions and connectors
            elif part.isalnum():
                # Distinguish between short and regular words
                if len(part) <= self.small_word_length:
                    ttype = TokenType.SMALLWORD  # Articles, prepositions
                else:
                    ttype = TokenType.WORD  # Regular words
            elif re.fullmatch(r'[.!?,;:]+', part):
                ttype = TokenType.PUNCTUATION  # Sentence punctuation
            else:
                ttype = TokenType.OTHER  # Other symbols

            tokens.append(Token(
                seq=tid,
                token_type=ttype,
                length=len(part),
                char_start=char_pos
            ))
            char_pos += len(part)

        return tokens

    def _calculate_separation_ranks(self, tokens: List[Token]) -> None:
        """
        Assign break desirability ranks to each token based on linguistic patterns.

        This method analyzes token boundaries and assigns numerical ranks
        indicating how desirable each token position is for line breaks. Higher
        ranks indicate better break points from a readability perspective.

        The ranking system considers:
        - Punctuation boundaries (highest priority)
        - Break word positions (high priority)
        - Word boundaries (medium priority)
        - Token type transitions (variable priority)

        Args:
            tokens: List of Token objects to modify in-place with separation ranks
        """
        # Mapping of token type transitions to break desirability ranks
        # Higher values = more desirable break points
        RankKey = Tuple[Optional[TokenType], Optional[TokenType]]
        rank_map: Dict[RankKey, int] = {
            # Highest priority: after sentence-ending punctuation
            (TokenType.PUNCTUATION, None): 150,

            # High priority: before break words (natural pause points)
            (None, TokenType.BREAKWORD): 130,

            # Standard priority: between full words
            (TokenType.WORD, TokenType.WORD): 100,

            # Medium priority: after full words
            (TokenType.WORD, None): 50,

            # Lower priority: small word or other token before full word
            (TokenType.SMALLWORD, TokenType.WORD): 30,
            (TokenType.OTHER, TokenType.WORD): 30,

            # Low priority: within small word or symbol clusters
            (TokenType.OTHER, TokenType.OTHER): 10,
            (TokenType.SMALLWORD, TokenType.SMALLWORD): 20,
            (TokenType.SMALLWORD, TokenType.OTHER): 20,
            (TokenType.OTHER, TokenType.SMALLWORD): 20,
        }

        for idx, tok in enumerate(tokens):
            rank = 0

            if tok.token_type == TokenType.WHITESPACE:
                # For whitespace, rank based on surrounding non-whitespace tokens
                prev_t = self._prev_non_ws(tokens, idx)
                next_t = self._next_non_ws(tokens, idx)
                ptype = prev_t.token_type if prev_t else None
                ntype = next_t.token_type if next_t else None

                # Try exact match first, then fallback to partial matches
                rank = rank_map.get((ptype, ntype),
                       rank_map.get((ptype, None),
                       rank_map.get((None, ntype), 0)))

            elif tok.token_type == TokenType.PUNCTUATION:
                # Allow breaks after punctuation even without explicit whitespace
                if not self._has_space_around(tokens, idx):
                    rank = 100

            tok.separation_rank = rank

    def _prev_non_ws(self, tokens: List[Token], idx: int) -> Optional[Token]:
        """Find the nearest non-whitespace token before the given index."""
        i = idx - 1
        while i >= 0:
            if tokens[i].token_type != TokenType.WHITESPACE:
                return tokens[i]
            i -= 1
        return None

    def _next_non_ws(self, tokens: List[Token], idx: int) -> Optional[Token]:
        """Find the nearest non-whitespace token after the given index."""
        i = idx + 1
        while i < len(tokens):
            if tokens[i].token_type != TokenType.WHITESPACE:
                return tokens[i]
            i += 1
        return None

    def _has_space_around(self, tokens: List[Token], idx: int) -> bool:
        """Check if a token is adjacent to whitespace on either side."""
        left = tokens[idx - 1] if idx > 0 else None
        right = tokens[idx + 1] if idx < len(tokens) - 1 else None

        return ((left and left.token_type == TokenType.WHITESPACE)
                or (right and right.token_type == TokenType.WHITESPACE))

    def _find_break_points(self, tokens: List[Token]) -> List[int]:
        """
        Determine optimal line break positions using efficient token-based algorithm.

        This method implements a linear-scan algorithm that works with tokens
        to find optimal break points.

        Args:
            tokens: List of Token objects with calculated separation ranks

        Returns:
            Sorted list of character positions where new lines should begin.
            Always includes position 0 (start of text).
        """
        # Phase 1: Filter tokens that can serve as break points
        # Only tokens with positive separation ranks are considered viable
        break_candidates = [
            token for token in tokens
            if token.separation_rank > 0
        ]

        # Edge case: no valid break candidates found
        if not break_candidates:
            return [0]  # Return single break at text start

        # Phase 2: Calculate text metrics and algorithm parameters
        last_token = tokens[-1]
        total_length = last_token.char_start + last_token.length
        num_lines = max(1, math.ceil(total_length / self.line_length))
        target_line_length = self.line_length
        tolerance_chars = math.ceil(target_line_length * self.tolerance)

        # Calculate minimum line lengths for meaningful line breaks
        # A line break only makes sense if both resulting lines meet minimum length
        min_line_length_base = target_line_length // 2  # At least half target length
        min_line_length = max(tolerance_chars, min_line_length_base)  # Use larger value

        # Distance penalty multiplier for scoring candidates
        # Higher balanced values create stronger penalties for distance from target
        base_penalty = 3.0 + (self.balanced * 2.0) if self.balanced > 0 else 0.5

        # Initialize result list and tracking variables
        breaks = [0]  # Always start with position 0
        start_idx = 0  # Current position in break_candidates list

        # Phase 3: Main loop - find break point for each line
        while len(breaks) < num_lines and start_idx < len(break_candidates):
            # Calculate positions for current line
            current_line_start = breaks[-1]
            target_position = current_line_start + target_line_length  # Ideal break position
            min_position = target_position - tolerance_chars  # Start of tolerance window
            max_position = target_position + tolerance_chars  # End of tolerance window

            # Early termination checks for minimum line lengths
            # Check 1: Would the remaining text after target position be too short?
            remaining_after_target = total_length - target_position
            if remaining_after_target < min_line_length:
                # Final line would be too short, don't create another break
                break

            # Check 2: Would the current line (up to target) be too short?
            current_line_length = target_position - current_line_start
            if current_line_length < min_line_length and len(breaks) > 1:
                # Current line would be too short, don't create break here
                # (Skip this check for first line as it starts at position 0)
                break

            # Phase 3a: Categorize candidates into preferred window vs fallback
            window_candidates = []    # Candidates within tolerance window (preferred)
            fallback_candidates = []  # Candidates outside window (fallback only)

            # Scan remaining candidates starting from current position
            for i in range(start_idx, len(break_candidates)):
                token = break_candidates[i]

                # Skip candidates that would create current line too short
                # This prevents breaking too early in the current line
                if token.char_start - current_line_start < min_line_length:
                    continue

                # Additional check: ensure remaining text after this candidate would be long enough
                remaining_after_candidate = total_length - token.char_start
                if remaining_after_candidate < min_line_length:
                    # This candidate would leave final line too short, skip it
                    continue

                # Categorize candidate based on position relative to tolerance window
                if min_position <= token.char_start <= max_position:
                    # Preferred: candidate is within tolerance window
                    window_candidates.append((i, token))
                else:
                    # Fallback: candidate is outside window but still viable
                    fallback_candidates.append((i, token))

                # Early termination optimization: if we have candidates and current
                # token is beyond reasonable distance, stop scanning
                if (window_candidates or fallback_candidates) and \
                        token.char_start > max_position + tolerance_chars:
                    break

            # Phase 3b: Select candidate pool (prefer window candidates)
            # Window-first strategy: use window candidates if available,
            # otherwise fall back to all candidates with higher penalty
            candidates = window_candidates if window_candidates else fallback_candidates

            # If no candidates found, we cannot create more lines
            if not candidates:
                break

            # Phase 3c: Score candidates and select best
            # Apply different penalty factors based on candidate source
            penalty_factor = base_penalty if window_candidates else base_penalty * 5

            def score_candidate(idx_token_pair):
                """
                Calculate candidate score: separation_rank - distance_penalty

                Higher separation_rank = better linguistic break point
                Lower distance from target = better line length balance
                """
                _, token = idx_token_pair
                distance = abs(token.char_start - target_position)
                return token.separation_rank - distance * penalty_factor

            # Select candidate with highest score
            best_idx, best_token = max(candidates, key=score_candidate)
            breaks.append(best_token.char_start)

            # Advance to next unprocessed candidate for efficiency
            # This ensures we don't re-evaluate candidates in future iterations
            start_idx = best_idx + 1

        return breaks
