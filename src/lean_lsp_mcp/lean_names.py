"""Lean identifier syntax as regex fragments.

Port of `isLetterLike`, `isSubScriptAlnum`, `isIdFirst` and `isIdRest` from
Lean's `Init/Meta/Defs.lean` (v4.34.1). Shared by theorem-name validation and
by source matching, so both agree on where a Lean identifier ends.
"""

import re

_LETTER_LIKE = (
    "\u03b1-\u03ba\u03bc-\u03c9"  # lower Greek, but λ
    "\u0391-\u039f\u03a1\u03a2\u03a4-\u03a9"  # upper Greek, but Π and Σ
    "\u03ca-\u03fb"  # Coptic letters
    "\u1f00-\u1ffe"  # polytonic Greek
    "\u2100-\u214f"  # letter-like block
    "\U0001d49c-\U0001d59f"  # script, double-struck, Fraktur
    "\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff"  # Latin-1 letters, but × and ÷
    "\u0100-\u017f"  # Latin Extended-A
)
_SUBSCRIPT_ALNUM = "\u2080-\u2089\u2090-\u209c\u1d62-\u1d6a\u2c7c"

# Characters that may start / continue a Lean identifier.
LEAN_ID_FIRST = f"[A-Za-z_{_LETTER_LIKE}]"
LEAN_ID_REST = f"[A-Za-z0-9_'!?{_LETTER_LIKE}{_SUBSCRIPT_ALNUM}]"

_LEAN_ID = f"{LEAN_ID_FIRST}{LEAN_ID_REST}*"
# A dot-separated name of plain identifiers, such as `Nat.add_comm`.
# `«...»` escaped components are not matched.
LEAN_NAME_RE = re.compile(rf"{_LEAN_ID}(?:\.{_LEAN_ID})*")
