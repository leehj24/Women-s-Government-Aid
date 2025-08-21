# textprep.py
import re
import pandas as pd

def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = re.sub(r"(?m)^\s*([\-–—\u2010-\u2015\u2212\u2043\u2022\u25CB\u25CF\u25AA\u25A0\u30FB]|[0-9]+[.)])\s+", " ", t)
    t = re.sub(r"[\u2460-\u2473\u3251-\u325F\u32B1-\u32BF]", " ", t)
    t = re.sub(r"[※＊*]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _clean_field_value(val: str, field_name: str) -> str:
    if not isinstance(val, str): return ""
    v = re.sub(rf"^\s*{re.escape(field_name)}\s*[:\-–—]?\s*", "", val.strip())
    return _normalize_text(v)

def unify_text_natural(row: pd.Series) -> str:
    title   = _normalize_text(row.get("제목",""))
    region  = _normalize_text(row.get("지역",""))
    target  = _clean_field_value(row.get("지원대상",""), "지원대상")
    content = _clean_field_value(row.get("지원내용",""), "지원내용")
    parts = []
    if title:   parts.append(f"정책명은 {title}")
    if region:  parts.append(f"지역은 {region}")
    if target:  parts.append(f"지원대상은 {target}")
    if content: parts.append(f"지원내용은 {content}")
    if not parts: return ""
    if len(parts) == 1:  return parts[0] + "이다."
    if len(parts) == 2:  return parts[0] + "이고, " + parts[1] + "이다."
    return "이고, ".join(parts[:-1]) + "이며, " + parts[-1] + "이다."

def ensure_text_columns(df: pd.DataFrame, text_col_name="검색본문_nat") -> pd.DataFrame:
    if text_col_name not in df.columns:
        df[text_col_name] = df.apply(unify_text_natural, axis=1)
    return df
