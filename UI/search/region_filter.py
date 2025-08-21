# region_filter.py
import re
from typing import Optional, Tuple

TOP_SYNONYM_TO_STD = {
    "서울":"서울특별시", "서울시":"서울특별자치시" if False else "서울특별시", "서울특별시":"서울특별시",
    "부산":"부산광역시", "부산시":"부산광역시", "부산광역시":"부산광역시",
    "대구":"대구광역시", "대구시":"대구광역시", "대구광역시":"대구광역시",
    "인천":"인천광역시", "인천시":"인천광역시", "인천광역시":"인천광역시",
    "광주":"광주광역시", "광주시":"광주광역시", "광주광역시":"광주광역시",
    "대전":"대전광역시", "대전시":"대전광역시", "대전광역시":"대전광역시",
    "울산":"울산광역시", "울산시":"울산광역시", "울산광역시":"울산광역시",
    "세종":"세종특별자치시", "세종시":"세종특별자치시", "세종특별자치시":"세종특별자치시",
    "경기":"경기도", "경기도":"경기도",
    "강원":"강원특별자치도", "강원특별자치도":"강원특별자치도",
    "충북":"충청북도", "충청북도":"충청북도",
    "충남":"충청남도", "충청남도":"충청남도",
    "전북":"전북특별자치도", "전북특별자치도":"전북특별자치도",
    "전남":"전라남도", "전라남도":"전라남도",
    "경북":"경상북도", "경상북도":"경상북도",
    "경남":"경상남도", "경상남도":"경상남도",
    "제주":"제주특별자치도", "제주특별자치도":"제주특별자치도",
}

TOP_TOKEN_PATTERN = "|".join(
    sorted(map(re.escape, TOP_SYNONYM_TO_STD.keys()), key=len, reverse=True)
)

def _norm_region_value(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    return s

def extract_top_sub(query: str) -> Tuple[Optional[str], Optional[str]]:
    q = (query or "").strip()
    m = re.search(
        rf"({TOP_TOKEN_PATTERN})(?:\s*[\/\s]\s*([가-힣]+(?:군|구|시|읍|면|동)))?",
        q
    )
    if not m:
        return None, None
    top_raw = m.group(1)
    sub = m.group(2) if len(m.groups()) >= 2 else None
    top_std = TOP_SYNONYM_TO_STD.get(top_raw, top_raw)
    return top_std, sub

def region_pass(policy_region: str, top_std: Optional[str], sub: Optional[str]) -> bool:
    # 지역 신호가 없으면 필터 미적용
    if not top_std and not sub:
        return True
    s = _norm_region_value(policy_region)
    if sub:
        return s.startswith(f"{top_std}/{sub}")
    if s.startswith(top_std): return True
    if s.startswith(top_std + "("): return True
    if (top_std in s) and ("교육청" in s or "교육지원청" in s or "지원청" in s): return True
    return False
