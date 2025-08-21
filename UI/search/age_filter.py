# age_filter.py
import json, re
from datetime import date
from dateutil import parser as dateparser
from typing import List, Tuple, Optional

AGE_RANGE = re.compile(r"(?:만\s*)?(\d{1,3})\s*[~\-]\s*(?:만\s*)?(\d{1,3})\s*세")
AGE_SINGLE = re.compile(r"(?:만\s*)?(\d{1,3})\s*세\s*(이상|초과|이하|미만)?")

def parse_query_age(text: str) -> List[Tuple[int,int]]:
    out: List[Tuple[int,int]] = []
    if not text: return out
    for a,b in AGE_RANGE.findall(text):
        a,b = int(a), int(b)
        lo,hi = min(a,b), max(a,b)
        out.append((lo,hi))
    for n,b in AGE_SINGLE.findall(text):
        n = int(n)
        if not b: out.append((n,n))
        elif b in ("이상","초과"): out.append((n + (b=="초과"), 200))
        elif b in ("이하","미만"): out.append((0, n - (b=="미만")))
    m = re.search(r"(\d{1,2})\s*대", text)
    if m:
        d = int(m.group(1)); out.append((d*10, d*10+9))
    return out

def _parse_birthdate(s: str):
    if not s: return None
    try:
        return dateparser.parse(s.strip()).date()
    except Exception:
        return None

def dob_to_range(dob_str: Optional[str]) -> List[Tuple[int,int]]:
    if not dob_str: return []
    b = _parse_birthdate(dob_str)
    if not b: return []
    today = date.today()
    age = int((today - b).days / 365.2425)
    age = max(0, min(200, age))
    return [(age, age)]

def age_match(policy_ranges_str: Optional[str], req_ranges: List[Tuple[int,int]]) -> bool:
    # 정책 JSON이 없으면 무제한으로 간주
    try:
        pr = json.loads(policy_ranges_str) if policy_ranges_str else []
    except Exception:
        pr = []
    if not pr:
        pr = [(0,200)]
    if not req_ranges:
        return True
    for lo1,hi1 in pr:
        for lo2,hi2 in req_ranges:
            if max(lo1,lo2) <= min(hi1,hi2):
                return True
    return False
