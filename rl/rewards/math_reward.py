"""Binary math-answer verification for RLVR.

`verify_math(completion, reference_answer)` extracts the completion's final
answer (last `\\boxed{...}`, GSM8K `#### x`, or the last number-bearing line),
normalizes both sides out of LaTeX into sympy-parseable text, and scores 1.0
iff they are equivalent:

  1. exact normalized-string match (also covers non-scalar answers like
     intervals/tuples that sympy comparison would refuse),
  2. sympy equivalence (`Expr.equals`, then `simplify(a - b) == 0`) — handles
     fractions vs decimals, algebraic rearrangements, radicals, percents,
  3. float comparison fallback (rel 1e-6) when sympy cannot parse.

Fractions (`\\frac`, `a/b`), percents (`50%` == 1/2), comma-grouped numbers
(`1,234`), `$` amounts, `\\sqrt`, `\\pi`, `^` powers, and `\\text{...}` units
are all normalized. `sympy` is imported lazily (function-local) so importing
this module — and everything in `G` that depends on it — stays instant.
"""

from __future__ import annotations

import math
import re

#: Answers longer than this are junk; refuse to normalize/parse them.
MAX_ANSWER_CHARS = 300

_ANSWER_PREFIX_RE = re.compile(r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:=]?\s*", re.IGNORECASE)
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_FRAC_DIGITS_RE = re.compile(r"\\[dt]?frac\s*(\d)\s*(\d)")
_SQRT_N_RE = re.compile(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SQRT_BARE_RE = re.compile(r"\\sqrt\s*(\d+(?:\.\d+)?)")
_TEXT_RE = re.compile(r"\\text(?:rm|bf|it|tt)?\s*\{[^{}]*\}")
_MATHRM_RE = re.compile(r"\\math(?:rm|bf|it|frak|cal|bb)\s*\{([^{}]*)\}")
_DEGREE_RE = re.compile(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|°")
_COMMA_IN_NUMBER_RE = re.compile(r"(?<=\d),(?=\d)")
_PERCENT_RE = re.compile(r"^\(?(-?\d+(?:\.\d+)?)\)?\s*%$")
_GSM8K_HASH_RE = re.compile(r"####\s*([^\n]+)")
_NUMBER_TOKEN_RE = re.compile(r"-?\s?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?")
_SIMPLE_FRACTION_RE = re.compile(r"\(?(-?\d+(?:\.\d+)?)\)?\s*/\s*\(?(-?\d+(?:\.\d+)?)\)?")
_BIG_EXPONENT_RE = re.compile(r"\*\*\s*\(?\s*-?\d{4,}")
_BIG_NUMBER_RE = re.compile(r"\d{40,}")


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #


def extract_boxed(text: str) -> str | None:
    """Content of the LAST `\\boxed{...}` / `\\fbox{...}` (balanced braces);
    also supports the brace-less `\\boxed 5` form. None if absent/empty."""
    candidates = [(text.rfind(marker), marker) for marker in ("\\boxed", "\\fbox")]
    pos, marker = max(candidates)
    if pos < 0:
        return None
    i = pos + len(marker)
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return None
    if text[i] == "{":
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j].strip() or None
        return None  # unbalanced braces
    token = re.match(r"[^\s${}]+", text[i:])
    return token.group(0) if token else None


def _extract_hash_answer(text: str) -> str | None:
    """GSM8K-style final line: the text after the last `####`."""
    found = _GSM8K_HASH_RE.findall(text)
    if found:
        return found[-1].strip() or None
    return None


def _extract_last_number(text: str) -> str | None:
    """Last number-like token (int/decimal/fraction/percent, `$`/comma-tolerant)
    on the last line that contains one."""
    for line in reversed(text.strip().splitlines()):
        found = _NUMBER_TOKEN_RE.findall(line)
        if found:
            return found[-1].replace("$", "").replace(" ", "")
    return None


def extract_final_answer(completion: str) -> str | None:
    """Model completion -> raw final-answer string (None if nothing found).

    Priority: last `\\boxed{...}` > last `#### x` > last number line.
    """
    if not completion or not completion.strip():
        return None
    return extract_boxed(completion) or _extract_hash_answer(completion) or _extract_last_number(completion)


# --------------------------------------------------------------------------- #
# Normalization: LaTeX answer text -> sympy-parseable expression string
# --------------------------------------------------------------------------- #


def normalize_answer(answer: str) -> str:
    """Canonicalize an extracted answer; returns '' if it is unusable."""
    s = answer.strip()
    if not s or len(s) > MAX_ANSWER_CHARS:
        return ""
    s = _ANSWER_PREFIX_RE.sub("", s).strip()
    for token in ("$$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(token, "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\;", "").replace("\\,", "").replace("\\ ", " ")
    s = _TEXT_RE.sub("", s)
    s = _MATHRM_RE.sub(r"\1", s)
    s = s.replace("\\%", "%").replace("\\$", "").replace("$", "")
    s = s.replace("\\boxed", "").replace("\\fbox", "")
    for _ in range(8):  # innermost-out, so nested \frac resolves
        new = _FRAC_RE.sub(r"(\1)/(\2)", s)
        new = _FRAC_DIGITS_RE.sub(r"(\1)/(\2)", new)
        if new == s:
            break
        s = new
    s = _SQRT_N_RE.sub(r"((\2))**(1/(\1))", s)
    for _ in range(4):
        new = _SQRT_RE.sub(r"sqrt(\1)", s)
        if new == s:
            break
        s = new
    s = _SQRT_BARE_RE.sub(r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\infty", "oo")
    s = _DEGREE_RE.sub("", s)
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = _COMMA_IN_NUMBER_RE.sub("", s)
    if "=" in s:  # "x = 5" -> "5"
        s = s.split("=")[-1]
    s = s.strip().strip(";").rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    percent = _PERCENT_RE.match(s)
    if percent:
        s = f"({percent.group(1)})/100"
    return s


# --------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------- #


def _looks_dangerous(s: str) -> bool:
    """Refuse sympy evaluation of expressions that could blow up (huge
    exponents / absurd literals); such strings fall back to string compare."""
    return bool(_BIG_EXPONENT_RE.search(s) or _BIG_NUMBER_RE.search(s) or s.count("**") > 3)


def _sympy_equal(a: str, b: str) -> bool | None:
    """True/False if sympy can decide equivalence, None if it cannot."""
    if _looks_dangerous(a) or _looks_dangerous(b):
        return None
    import sympy  # noqa: PLC0415 — lazy: [post] extra, keeps module import instant
    from sympy.parsing.sympy_parser import (  # noqa: PLC0415
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        expr_a = parse_expr(a, transformations=transformations)
        expr_b = parse_expr(b, transformations=transformations)
    except Exception:  # noqa: BLE001 — any parse failure means "undecidable"
        return None
    try:
        decided = expr_a.equals(expr_b)  # numeric probing; robust for constants
        if decided is not None:
            return bool(decided)
    except Exception:  # noqa: BLE001
        pass
    try:
        return bool(sympy.simplify(expr_a - expr_b) == 0)
    except Exception:  # noqa: BLE001
        return None


def _float_of(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        pass
    fraction = _SIMPLE_FRACTION_RE.fullmatch(s)
    if fraction:
        denominator = float(fraction.group(2))
        if denominator != 0.0:
            return float(fraction.group(1)) / denominator
    return None


def verify_math(completion: str, reference_answer: str) -> float:
    """1.0 iff the completion's final answer is equivalent to the reference.

    The reference may be a bare answer ("72", "\\frac{1}{2}") or carry its own
    `\\boxed{...}` / `#### x` wrapper (full GSM8K/MATH solutions work).
    """
    predicted = extract_final_answer(completion)
    if predicted is None:
        return 0.0
    reference = extract_boxed(reference_answer) or _extract_hash_answer(reference_answer) or reference_answer
    norm_pred = normalize_answer(predicted)
    norm_ref = normalize_answer(reference)
    if not norm_pred or not norm_ref:
        return 0.0
    if norm_pred == norm_ref:
        return 1.0
    decided = _sympy_equal(norm_pred, norm_ref)
    if decided is not None:
        return 1.0 if decided else 0.0
    float_pred, float_ref = _float_of(norm_pred), _float_of(norm_ref)
    if float_pred is not None and float_ref is not None:
        return 1.0 if math.isclose(float_pred, float_ref, rel_tol=1e-6, abs_tol=1e-9) else 0.0
    return 0.0


__all__ = [
    "MAX_ANSWER_CHARS",
    "extract_boxed",
    "extract_final_answer",
    "normalize_answer",
    "verify_math",
]
