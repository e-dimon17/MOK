"""Step G: math reward verifier — extraction, normalization, truth table."""

from __future__ import annotations

import pytest

from G.rewards.math_reward import (
    extract_boxed,
    extract_final_answer,
    normalize_answer,
    verify_math,
)

# --------------------------------------------------------------------------- #
# Truth table: (completion, reference, expected reward)
# --------------------------------------------------------------------------- #

TRUTH_TABLE = [
    # boxed fractions & decimal equivalence
    ("The answer is \\boxed{\\frac{1}{2}}.", "0.5", 1.0),
    ("\\boxed{0.5}", "\\frac{1}{2}", 1.0),
    ("\\boxed{-\\frac{3}{4}}", "-0.75", 1.0),
    ("\\boxed{\\frac{\\frac{1}{2}}{3}}", "1/6", 1.0),          # nested \frac
    ("\\boxed{1/3}", "0.5", 0.0),                                # wrong fraction
    # plain ints, GSM8K ####, comma/dollar numbers
    ("thus \\boxed{42}", "42", 1.0),
    ("blah blah\n#### 72", "72", 1.0),
    ("Total is 1,234 dollars", "1234", 1.0),
    ("The bill is $1,234.50 total", "1234.5", 1.0),
    # percent forms
    ("So the discount is 50%", "1/2", 1.0),
    ("\\boxed{50\\%}", "0.5", 1.0),
    # symbolic equivalence
    ("\\boxed{2x + 2}", "2(x+1)", 1.0),
    ("\\boxed{\\sqrt{8}}", "2\\sqrt{2}", 1.0),
    ("area is \\boxed{45^\\circ}", "45", 1.0),
    ("\\boxed{3.140}", "3.14", 1.0),
    ("x = -5", "-5", 1.0),
    # wrong answers
    ("The answer is 41.", "42", 0.0),
    ("\\boxed{\\frac{22}{7}}", "pi", 0.0),                       # close but not pi
    ("First 12 apples.\nThen total 24.", "12", 0.0),             # last line wins
    ("First 12 apples.\nThen total 24.", "24", 1.0),
    # malformed / empty
    ("", "42", 0.0),
    ("No numeric content here.", "7", 0.0),
    ("\\boxed{}", "5", 0.0),
    ("\\boxed{\\frac{1}{0}}", "5", 0.0),                         # division by zero
    # reference may carry its own wrapper
    ("The answer is \\boxed{72}", "\\boxed{72}", 1.0),
    ("I get 8 in the end", "Working shown here\n#### 8", 1.0),
]


@pytest.mark.parametrize(("completion", "reference", "expected"), TRUTH_TABLE)
def test_truth_table(completion: str, reference: str, expected: float) -> None:
    assert verify_math(completion, reference) == expected


def test_reward_is_binary_float() -> None:
    for completion, reference, _ in TRUTH_TABLE:
        reward = verify_math(completion, reference)
        assert isinstance(reward, float)
        assert reward in (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_extract_boxed_balanced_and_last() -> None:
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"     # last wins
    assert extract_boxed("\\fbox{7}") == "7"
    assert extract_boxed("\\boxed 5") == "5"                       # brace-less form
    assert extract_boxed("\\boxed{unclosed") is None
    assert extract_boxed("\\boxed{}") is None
    assert extract_boxed("nothing here") is None


def test_extract_final_answer_priority() -> None:
    # boxed beats #### beats last-number-line
    assert extract_final_answer("#### 9 and \\boxed{3}") == "3"
    assert extract_final_answer("count 4\n#### 9") == "9"
    assert extract_final_answer("count 4\nthen 6 total") == "6"
    assert extract_final_answer("") is None
    assert extract_final_answer("words only") is None


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def test_normalize_answer_latex_forms() -> None:
    assert normalize_answer("\\frac{3}{4}") == "(3)/(4)"
    assert normalize_answer("$1,234$") == "1234"
    assert normalize_answer("50\\%") == "(50)/100"
    assert normalize_answer("x = 7") == "7"
    assert normalize_answer("\\sqrt{2}") == "sqrt(2)"
    assert normalize_answer("2^{10}") == "2**(10)"
    assert normalize_answer("12\\text{ cm}") == "12"
    assert normalize_answer("") == ""
    assert normalize_answer("x" * 500) == ""  # over MAX_ANSWER_CHARS
