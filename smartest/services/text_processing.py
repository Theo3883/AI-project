from __future__ import annotations

import re
import unicodedata
from typing import Dict, List


class TextProcessor:
    WORD_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = self._strip_accents(text)
        tokens = self.WORD_RE.findall(text)
        lemmas = [self._pseudo_lemma(t) for t in tokens]
        return " ".join(lemmas)

    def _strip_accents(self, text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def _pseudo_lemma(self, token: str) -> str:
        for suf in ("lor", "urilor", "ilor", "ului", "are", "ul", "le", "ii", "i", "e"):
            if token.endswith(suf) and len(token) > len(suf) + 2:
                return token[: -len(suf)]
        return token

    def keyword_score(self, norm_correct: str, norm_user: str) -> float:
        set_c = set(norm_correct.split())
        set_u = set(norm_user.split())
        if not set_c:
            return 0.0
        inter = len(set_c & set_u)
        union = len(set_c | set_u)
        if union == 0:
            return 0.0
        jaccard = inter / union
        return 100.0 * jaccard

    def extract_assignments(self, norm_text: str) -> Dict[str, str]:
        pairs: Dict[str, str] = {}
        for part in norm_text.replace(",", " ").split():
            if "=" in part:
                var, val = part.split("=", 1)
                pairs[var.strip()] = val.strip()
        return pairs

    def extract_numbers(self, norm_text: str) -> List[int]:
        return [int(x) for x in re.findall(r"-?[0-9]+", norm_text)]
