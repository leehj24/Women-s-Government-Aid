# label_filter.py
import re
from typing import List

_norm_label_re = re.compile(r"[\s/()\[\]{}·.,\-]+")

def _norm_label(s: str) -> str:
    return _norm_label_re.sub("", s or "").lower()

def _expand_queries(queries: List[str]) -> List[str]:
    out = set()
    for q in queries or []:
        q = (str(q) if q is not None else "").strip()
        if not q: continue
        out.add(q)
        for tok in re.split(r"[\/,|]", q):
            tok = tok.strip()
            if tok: out.add(tok)
        no_paren_chars    = re.sub(r"[()]", "", q)
        no_paren_content  = re.sub(r"\([^)]*\)", "", q)
        out.update([no_paren_chars, no_paren_content])
        def _nz(s): return _norm_label(s) if s else s
        out.update({_nz(q), _nz(no_paren_chars), _nz(no_paren_content)})
        SYN = {
            "대학(원)생": ["대학생", "대학원생"],
            "한부모": ["한부모가족", "한부모가정"],
            "고령자": ["노인", "어르신"],
            "1인가구": ["1인 가구", "독거"],
            "임신/출산/육아": ["임신", "출산", "육아", "임신출산육아"],
        }
        if q in SYN:
            out.update(SYN[q]); out.update(_nz(x) for x in SYN[q])
    return [x for x in out if x]

def contains_any(cell: str, queries: List[str]) -> bool:
    s = str(cell) if cell is not None else ""
    ex = _expand_queries(queries)
    if any(q for q in ex):
        if any(q in s for q in ex):
            return True
    s_norm = _norm_label(s)
    return any((qn and (qn in s_norm)) for qn in ex)
