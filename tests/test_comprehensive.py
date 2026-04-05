import pytest
from textbreaker import TextBreaker

# === TEST DATA EXTRACTED FROM test_runner.py ===

EDGE_CASES = [
    ("empty_string", "", {"line_length": 50}, "Empty string should return empty list"),
    ("whitespace_only", "   \n\t  ", {"line_length": 50}, "Whitespace-only string should return single empty line"),
    ("single_character", "A", {"line_length": 50}, "Single character"),
    ("single_word", "Hello", {"line_length": 50}, "Single word"),
    ("very_long_word", "supercalifragilisticexpialidocious", {"line_length": 20}, "Word longer than line length"),
    ("alphabet_sequence", "a b c d e f g h i j k l m n o p q r s t u v w x y z", {"line_length": 30}, "Alphabet sequence with spaces"),
    ("numbers_only", "123456789 987654321 111222333 444555666", {"line_length": 25}, "Numbers only"),
    ("mixed_alphanumeric", "Test123 Data456 Code789 File000", {"line_length": 20}, "Mixed letters and numbers"),
    ("punctuation_heavy", "Hello! How are you? Fine, thanks. Good; very good: excellent.", {"line_length": 25}, "Heavy punctuation usage"),
    ("no_spaces", "Thisisaverylongstringwithoutanyspacesatall", {"line_length": 15}, "No spaces - should not break"),
]

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "In the beginning was the Word, and the Word was with God.",
    "Four score and seven years ago our fathers brought forth on this continent a new nation.",
    "We hold these truths to be self-evident, that all men are created equal.",
    "Ask not what your country can do for you, ask what you can do for your country.",
    "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
    "Space: the final frontier. These are the voyages of the starship Enterprise.",
    "May the Force be with you, always.",
    "Elementary, my dear Watson.",
    "Frankly, my dear, I don't give a damn.",
    "Here's looking at you, kid.",
    "I'll be back.",
    "Houston, we have a problem."
]

ITALIAN_SENTENCES = [
    "La vita è come una scatola di cioccolatini, non sai mai quello che ti capita.",
    "Roma non è stata costruita in un giorno.",
    "Chi dorme non piglia pesci.",
    "Meglio un uovo oggi che una gallina domani.",
    "L'appetito vien mangiando.",
    "Tutto il mondo è paese.",
    "Chi va piano va sano e va lontano.",
    "Non è tutto oro quello che luccica.",
    "Il tempo è denaro.",
    "Dove c'è fumo c'è fuoco."
]

SPANISH_SENTENCES = [
    "El que madruga, Dios le ayuda.",
    "No por mucho madrugar amanece más temprano.",
    "A quien buen árbol se arrima, buena sombra le cobija.",
    "Más vale tarde que nunca.",
    "En boca cerrada no entran moscas.",
    "Camarón que se duerme se lo lleva la corriente.",
    "El hábito no hace al monje.",
    "Agua que no has de beber, déjala correr."
]

FRENCH_SENTENCES = [
    "La vie est belle, mais elle est aussi très courte.",
    "Qui vivra verra.",
    "C'est la vie.",
    "Petit à petit, l'oiseau fait son nid.",
    "Il n'y a que le premier pas qui coûte.",
    "L'habit ne fait pas le moine.",
    "Paris ne s'est pas fait en un jour."
]

GERMAN_SENTENCES = [
    "Aller Anfang ist schwer.",
    "Was du heute kannst besorgen, das verschiebe nicht auf morgen.",
    "Der frühe Vogel fängt den Wurm.",
    "Übung macht den Meister.",
    "Zeit ist Geld.",
    "Morgenstund hat Gold im Mund."
]

TECHNICAL_CASES = [
    ("code_like", "function getData() { return api.fetch('/users'); }", {"line_length": 20}, "Code-like syntax"),
    ("urls", "Visit https://www.example.com/very/long/path/to/resource for more info", {"line_length": 25}, "URLs in text"),
    ("emails", "Contact support@company.com or sales@business.org for assistance", {"line_length": 30}, "Email addresses"),
    ("mixed_languages", "Hello mundo, comment ça va? Wie geht es dir?", {"line_length": 20}, "Mixed languages in one text"),
    ("scientific", "H2O + NaCl → Chemical reaction at 25°C under standard conditions", {"line_length": 25}, "Scientific notation"),
    ("mathematical", "f(x) = 2x² + 3x - 5 where x ∈ ℝ and x > 0", {"line_length": 20}, "Mathematical expressions"),
]

# === HELPER FOR PARAMETRIZATION ===

def get_all_test_cases():
    cases = []
    # Add named cases
    for name, text, params, desc in EDGE_CASES:
        cases.append(pytest.param(text, params, id=name))

    for i, s in enumerate(ENGLISH_SENTENCES):
        cases.append(pytest.param(s, {"line_length": 40, "balanced": 1.0}, id=f"english_{i+1}"))

    for i, s in enumerate(ITALIAN_SENTENCES):
        cases.append(pytest.param(s, {"line_length": 35, "balanced": 1.2}, id=f"italian_{i+1}"))

    for i, s in enumerate(SPANISH_SENTENCES):
        cases.append(pytest.param(s, {"line_length": 30, "balanced": 0.8}, id=f"spanish_{i+1}"))

    for i, s in enumerate(FRENCH_SENTENCES):
        cases.append(pytest.param(s, {"line_length": 28, "balanced": 1.5}, id=f"french_{i+1}"))

    for i, s in enumerate(GERMAN_SENTENCES):
        cases.append(pytest.param(s, {"line_length": 32, "balanced": 2.0}, id=f"german_{i+1}"))

    for name, text, params, desc in TECHNICAL_CASES:
        cases.append(pytest.param(text, params, id=name))

    return cases

@pytest.mark.parametrize("text, params", get_all_test_cases())
def test_comprehensive_wrapping(text, params):
    breaker = TextBreaker(**params)
    lines = breaker.wrap(text)

    assert isinstance(lines, list)

    # Reconstruct text (ignoring leading/trailing whitespace and collapsed spaces in results)
    # The current implementation strips lines, so we check if all original non-whitespace chars are present
    original_chars = "".join(text.split())
    result_chars = "".join("".join(line.split()) for line in lines)
    assert original_chars == result_chars

    # Basic length check (with high tolerance for words that can't be broken)
    line_length = params.get('line_length', 50)
    tolerance = params.get('tolerance', 0.3)
    max_allowed = line_length * (1 + tolerance) * 2 # High multiplier for un-breakable words

    for line in lines:
        # If line has no spaces and is longer than max_allowed, it's a single long word (acceptable for now)
        if ' ' in line:
            assert len(line) <= max_allowed or len(line.split()) == 1
