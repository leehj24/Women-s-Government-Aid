# policy_search.py
# - 데이터 로드 & 엔진 초기화
# - find_policies / CLI만 제공(나머지는 모듈에 위임)
# - Scala 검색 엔진 우선 사용 (사용 불가시 Python 엔진으로 폴백)

import argparse
import pandas as pd
from typing import List, Optional, Literal
import logging
from datetime import datetime
from .loader import load_dataframe
from .engine import SearchEngine

log = logging.getLogger(__name__)

# 전역 싱글턴(간단하게)
_df_raw, _df, _path = load_dataframe()
_engine = SearchEngine(_df)

# Scala 검색 엔진 래퍼 (지연 로딩)
_scala_wrapper = None

def _get_scala_wrapper():
    """Scala 검색 엔진 래퍼 가져오기 (지연 초기화)"""
    global _scala_wrapper
    if _scala_wrapper is None:
        try:
            from .scala_search_wrapper import get_scala_wrapper
            _scala_wrapper = get_scala_wrapper()
        except Exception as e:
            log.warning(f"Scala search engine not available: {e}")
            _scala_wrapper = None
    return _scala_wrapper

def _calculate_age_from_dob(dob: str) -> Optional[int]:
    """생년월일(YYYY-MM-DD)에서 만나이 계산"""
    if not dob:
        return None
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    except Exception:
        return None

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
                  out: Literal["dataframe","json","csv"]="dataframe",
                  use_scala: bool = True):
    """
    - input이 비어 있으면: 추천(필터만) — 신청 마감 임박 우선 정렬
    - input이 있으면: 검색 (Scala 엔진 우선, 없으면 Python FAISS 엔진)
    - region="" 또는 "전국"이면 지역 제한 없음
    - dob="YYYY-MM-DD" 형식이면 만나이 계산하여 age_eff_ranges(JSON)과 교집합
    - use_scala: True면 Scala 엔진 우선 사용 (기본값: True)
    """
    # 나이 계산
    age = _calculate_age_from_dob(dob) if dob else None
    
    # Scala 엔진 사용 시도
    if use_scala:
        scala_wrapper = _get_scala_wrapper()
        if scala_wrapper and scala_wrapper.available:
            try:
                # 지역 정규화
                region_normalized = region if region and region != "전국" else None
                
                if not input or not str(input).strip():
                    # 추천 모드
                    df_scala = scala_wrapper.recommend(
                        region=region_normalized,
                        age=age,
                        categories=categories,
                        supports=supports,
                        top_k=topk or 200
                    )
                else:
                    # 검색 모드
                    df_scala = scala_wrapper.search(
                        query=input,
                        region=region_normalized,
                        age=age,
                        categories=categories,
                        supports=supports,
                        top_k=topk or 200
                    )
                
                # Scala 결과를 원본 DataFrame과 병합하여 전체 컬럼 포함
                if not df_scala.empty and "index" in df_scala.columns:
                    # orig_index 컬럼 추가
                    df_scala = df_scala.rename(columns={"index": "orig_index"})
                    
                    # 원본 데이터와 병합
                    df_result = _df_raw.iloc[df_scala["orig_index"].tolist()].copy()
                    df_result.insert(0, "score", df_scala["score"].values)
                    df_result.insert(0, "orig_index", df_scala["orig_index"].values)
                    
                    log.info(f"Scala engine returned {len(df_result)} results")
                    return _format_output(df_result, out)
                    
            except Exception as e:
                log.warning(f"Scala engine failed, falling back to Python engine: {e}")
                # 폴백: Python 엔진 사용
    
    # Python 엔진 사용 (폴백 또는 use_scala=False)
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
