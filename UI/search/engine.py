# engine.py
import re, time, logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    REGION_COL, THRESHOLD, MODEL_NAME, TEXT_COLS, REGION_MODE, TOPK_CANDIDATES, CACHE_DIR
)
from .index_faiss import FaissIndex
from .age_filter import parse_query_age, dob_to_range, age_match
from .region_filter import extract_top_sub, region_pass
from .label_filter import contains_any
from .query_enhancer import enhance_search_scores, split_query_into_keywords

log = logging.getLogger(__name__)

def _find_col(df_in: pd.DataFrame, prefer: List[str], fuzzy: List[str]) -> Optional[str]:
    def _norm(s): return re.sub(r"[\s_/()\-·.,]+", "", s or "").lower()
    for c in prefer:
        if c in df_in.columns: return c
    nmap = {c: _norm(c) for c in df_in.columns}
    want = [_norm(x) for x in prefer+fuzzy]
    for c, nc in nmap.items():
        if any(w in nc for w in want): return c
    return None

def _last_date_from_period(s: str):
    if not isinstance(s,str) or not s.strip(): return pd.NaT
    s2 = s.replace("~","-").replace("–","-").replace("—","-")
    parts = re.findall(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", s2)
    try:
        import dateutil.parser as dp
        return dp.parse(parts[-1]) if parts else pd.NaT
    except Exception:
        return pd.NaT

class SearchEngine:
    def __init__(self, df: pd.DataFrame, cache_dir: str = CACHE_DIR,
                 text_cols: Tuple[str, ...] = TEXT_COLS, region_col: str = REGION_COL,
                 model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.df = df.reset_index(drop=True).copy()
        self.region_col = region_col
        self.texts = (self.df[list(text_cols)].fillna("").agg(" ".join, axis=1)).tolist()
        self.index = FaissIndex(model_name, device=device)

        t0 = time.time()
        self.index.load_or_build(cache_dir, self.texts)
        log.info("[SearchEngine] ready in %.1fs", time.time()-t0)

    # 1단계 필터: 카테고리/지원형태/지역/나이
    def _stage1_mask(self, df_in: pd.DataFrame,
                     categories: Optional[List[str]],
                     supports: Optional[List[str]],
                     region: Optional[str],
                     dob: Optional[str],
                     kw_text: Optional[str]) -> np.ndarray:
        n = len(df_in)
        mask = np.ones(n, dtype=bool)

        # 카테고리/지원형태 (부분일치 OR)
        if categories:
            col = _find_col(df_in, ['카테고리_분류','category_label'], ['카테고리','category','label'])
            if col:
                mask &= df_in[col].apply(lambda x: contains_any(x, categories)).to_numpy()
        if supports:
            col = _find_col(df_in, ['지원형태_분류','support_label'], ['지원형태','support','label'])
            if col:
                mask &= df_in[col].apply(lambda x: contains_any(x, supports)).to_numpy()

        # 지역: region 우선 → 없으면 쿼리에서 추출
        top_std = sub = None
        r = (region or "").strip()
        if r and r != "전국":
            top_std, sub = extract_top_sub(r)
            if not top_std:  # "대구" 같은 단어만 들어와도 통과시키기
                top_std = r
        elif kw_text:
            top_std, sub = extract_top_sub(kw_text)

        if (top_std or sub) and (self.region_col in df_in.columns):
            mask &= df_in[self.region_col].apply(lambda s: region_pass(s, top_std, sub)).to_numpy()

        # 나이: DOB 우선 → 없으면 쿼리 숫자 파싱, age_eff_ranges 있을 때만 적용
        req_ranges: List[Tuple[int,int]] = dob_to_range(dob) if dob else []
        if (not req_ranges) and kw_text:
            req_ranges = parse_query_age(kw_text)
        if req_ranges and ("age_eff_ranges" in df_in.columns):
            mask &= df_in["age_eff_ranges"].apply(lambda s: age_match(s, req_ranges)).to_numpy()

        return mask

    def recommend(self, region: Optional[str], dob: Optional[str],
                  categories: Optional[List[str]], supports: Optional[List[str]]) -> pd.DataFrame:
        mask = self._stage1_mask(self.df, categories or [], supports or [], region, dob, kw_text=None)
        tmp = self.df[mask].copy()
        if "신청기간" in tmp.columns:
            tmp["_end"] = tmp["신청기간"].apply(_last_date_from_period)
            tmp = tmp.sort_values(by=["_end","제목"], ascending=[True,True]).drop(columns=["_end"])
        return tmp.reset_index(drop=True)

    def search(self, query: str, region: Optional[str], dob: Optional[str],
               categories: Optional[List[str]], supports: Optional[List[str]],
               threshold: float = THRESHOLD) -> pd.DataFrame:
        D, I = self.index.query(query, topk=None)
        order, scores = I, D

        # 쿼리에서 지역명 추출 (쿼리에 포함된 경우)
        keywords, query_region = split_query_into_keywords(query)
        # 쿼리에 지역명이 있으면 우선 사용
        effective_region = query_region if query_region else region

        # 쿼리 신호(나이/지역)를 필터에 전달
        mask = self._stage1_mask(self.df, categories or [], supports or [], effective_region, dob, kw_text=query)

        rows_yes = []
        for rank, idx in enumerate(order):
            sc = float(scores[rank])
            if sc < threshold: 
                continue
            if not mask[idx]:
                continue
            rows_yes.append((sc, idx))

        if not rows_yes:
            return self.df.head(0).copy()

        # 점수 보정: 키워드 매칭 및 지역명 보너스/감점 적용
        faiss_scores = np.array([s for s, _ in rows_yes])
        faiss_indices = np.array([i for _, i in rows_yes])
        
        enhanced_scores, enhanced_indices = enhance_search_scores(
            self.df, query, faiss_scores, faiss_indices
        )

        # 상위 K개 선택
        top_k = min(TOPK_CANDIDATES, len(enhanced_indices))
        sel_indices = enhanced_indices[:top_k]
        sel_scores = enhanced_scores[:top_k]
        
        out = self.df.iloc[list(sel_indices)].copy()
        out.insert(0, "score", sel_scores)
        return out.reset_index(drop=True)
