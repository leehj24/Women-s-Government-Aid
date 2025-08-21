# policy_search.py
# - 데이터 로드 & 엔진 초기화
# - find_policies / CLI만 제공(나머지는 모듈에 위임)

import argparse
import pandas as pd
from typing import List, Optional, Literal
from .loader import load_dataframe
from .engine import SearchEngine

# 전역 싱글턴(간단하게)
_df_raw, _df, _path = load_dataframe()
_engine = SearchEngine(_df)

def _format_output(df_out: pd.DataFrame, out: Literal["dataframe","json","csv"]):
    if out == "dataframe":
        return df_out
    if out == "json":
        keep = [c for c in [
            "orig_index","제목","지역",
            "카테고리_분류","category_label",
            "지원형태_분류","support_label",
            "지원형태","신청기간","신청방법","접수기관",
            "지원대상","지원내용","문의처","기타","detail_url","age_eff_ranges","score"
        ] if c in df_out.columns]
        return df_out[keep].to_json(force_ascii=False, orient="records", indent=2)
    if out == "csv":
        return df_out.to_csv(index=False)
    raise ValueError("out must be one of {'dataframe','json','csv'}")

def find_policies(input: str = "",
                  topk: Optional[int] = None,   # (사용 안 함: 내부에서 전체 쿼리)
                  region: Optional[str] = "전국",
                  dob: Optional[str] = None,
                  categories: Optional[List[str]] = None,
                  supports: Optional[List[str]] = None,
                  out: Literal["dataframe","json","csv"]="dataframe"):
    """
    - input이 비어 있으면: 추천(필터만) — 신청 마감 임박 우선 정렬
    - input이 있으면: FAISS 검색 + 필터 교집합(점수는 반환 컬럼에 'score')
    - region="" 또는 "전국"이면 지역 제한 없음
    - dob="YYYY-MM-DD" 형식이면 만나이 계산하여 age_eff_ranges(JSON)과 교집합
    """
    if not input or not str(input).strip():
        df = _engine.recommend(region=region, dob=dob, categories=categories, supports=supports)
        return _format_output(df, out)
    df = _engine.search(query=input, region=region, dob=dob, categories=categories, supports=supports)
    return _format_output(df, out)

# Flask 등에서 필요하면 원본 DF 제공
def get_base_df():
    return _df_raw.copy()

# --- CLI ---
def _parse_list(s: Optional[str]):
    if not s: return []
    import re
    return [x.strip() for x in re.split(r"[;,/|,]", s) if x.strip()]

def main(argv=None):
    ap = argparse.ArgumentParser(description="여성 정책 검색/추천 엔진(모듈형)")
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--region", type=str, default="전국")
    ap.add_argument("--dob", type=str, default=None)
    ap.add_argument("--categories", type=str, default=None)
    ap.add_argument("--supports", type=str, default=None)
    ap.add_argument("--out", type=str, choices=["dataframe","json","csv"], default="json")
    args = ap.parse_args(argv)

    res = find_policies(
        input=args.input, region=args.region, dob=args.dob,
        categories=_parse_list(args.categories), supports=_parse_list(args.supports),
        out=args.out
    )
    if isinstance(res, pd.DataFrame):
        print(res.to_csv(index=False, sep="\t"))
    else:
        print(res)

if __name__ == "__main__":
    main()
