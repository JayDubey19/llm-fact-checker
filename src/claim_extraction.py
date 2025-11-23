# src/claim_extraction.py

import spacy
from typing import List

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_claims(text: str) -> List[str]:
    """
    Extract multiple claims from text.
    Heuristic:
    - split into sentences
    - keep sentences with at least one verb and length >= 15 chars
    - deduplicate while preserving order
    """
    nlp = _get_nlp()
    doc = nlp(text)

    claims = []
    seen = set()

    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 15:
            continue
        # must contain at least one verb/aux to be a 'claim'
        if not any(tok.pos_ in {"VERB", "AUX"} for tok in sent):
            continue

        # simple dedupe
        norm = s.lower()
        if norm not in seen:
            seen.add(norm)
            claims.append(s)

    return claims

if __name__ == "__main__":
    txt = "The government launched a solar pump scheme and also announced free electricity to farmers. People are happy."
    print(extract_claims(txt))
