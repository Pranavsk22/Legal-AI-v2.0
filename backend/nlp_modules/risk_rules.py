# backend/nlp_modules/risk_rules.py
import re
from typing import List

# ⚠️ Regex patterns keyed by risk‑label
# Keys starting with "NO_" are flagged if the pattern is ABSENT.
# Other keys are flagged if the pattern is PRESENT.
RULES: dict[str, re.Pattern] = {
    "NO_TERMINATION":       re.compile(r"\b(terminat(e|ion)|vacat(e|ion))\b", re.I),
    "NO_GOV_LAW":           re.compile(r"\b(govern(ed|ing)\s+by|laws?\s+of|jurisdiction)\b", re.I),
    "NO_NOTICE":            re.compile(r"\bnotice\b", re.I),
    "NO_INDEMNITY":         re.compile(r"\bindemn(if(y|ication)|ity)\b", re.I),
    "NO_LIABILITY_LIMIT":   re.compile(r"\blimit(ation)?\s*(of|on)?\s*liability\b", re.I),
    "NO_STAMP_PAPER":       re.compile(r"\b(stamp\s*(paper|duty)|non-judicial|e-stamp)\b", re.I),
    "AUTO_RENEWAL":         re.compile(r"\b(auto(matic)?\s*renew(al)?|automatically\s+renew(s)?)\b", re.I),
    "UNLIMITED_LIABILITY":   re.compile(r"\b(unlimited\s+liability|liable\s+for\s+indirect\s+damages|consequential\s+damages\s+without\s+limit)\b", re.I),
}

def detect_risks(text: str) -> List[str]:
    """Return list of risk labels (missing or present risks)."""
    detected = []
    for key, pat in RULES.items():
        found = bool(pat.search(text))
        if key.startswith("NO_"):
            if not found:
                detected.append(key)
        else:
            if found:
                detected.append(key)
    return detected
