# backend/nlp_modules/risk_rules.py
import re
from typing import List, Dict

# ⚠️ Regex patterns keyed by risk‑label
RULES: dict[str, re.Pattern] = {
    "NO_TERMINATION":  re.compile(r"\bterminat(e|ion)\b", re.I),
    "NO_GOV_LAW":      re.compile(r"\bgovern(ed|ing)\s+by\b", re.I),
    "NO_NOTICE":       re.compile(r"\bnotice\s+period\b", re.I),
    "NO_INDEMNITY":    re.compile(r"\bindemnif(y|ication)\b", re.I),
}

def detect_risks(text: str) -> List[str]:
    """Return list of missing‑clause risk labels."""
    missing = []
    for key, pat in RULES.items():
        if not pat.search(text):
            missing.append(key)
    return missing
