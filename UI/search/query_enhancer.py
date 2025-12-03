# query_enhancer.py
# 검색 쿼리 개선: 키워드 분리, 지역명 추출, 점수 보정

import re
from typing import Tuple, Set, Optional, List
import pandas as pd
import numpy as np

# 지역명 패턴 (긴 것부터)
REGION_PATTERNS = [
    "서울특별시", "서울시", "서울",
    "부산광역시", "부산시", "부산",
    "대구광역시", "대구시", "대구",
    "인천광역시", "인천시", "인천",
    "광주광역시", "광주시", "광주",
    "대전광역시", "대전시", "대전",
    "울산광역시", "울산시", "울산",
    "세종특별자치시", "세종시", "세종",
    "경기도", "경기",
    "강원도", "강원",
    "충청북도", "충북",
    "충청남도", "충남",
    "전라북도", "전북",
    "전라남도", "전남",
    "경상북도", "경북",
    "경상남도", "경남",
    "제주특별자치도", "제주도", "제주"
]

def split_query_into_keywords(query: str) -> Tuple[Set[str], Optional[str]]:
    """쿼리를 키워드와 지역명으로 분리"""
    query_lower = query.lower().strip()
    
    # 지역명 찾기 (긴 것부터)
    found_region = None
    remaining_query = query_lower
    
    for pattern in sorted(REGION_PATTERNS, key=len, reverse=True):
        if pattern in remaining_query:
            found_region = pattern
            remaining_query = remaining_query.replace(pattern, " ").strip()
            break
    
    # 남은 키워드 추출 (1글자 제외)
    keywords = set(w for w in remaining_query.split() if len(w) > 1)
    
    return keywords, found_region

def calculate_keyword_match_score(keywords: Set[str], text: str) -> float:
    """키워드 매칭 점수 계산 (개선 버전)"""
    if not keywords:
        return 0.0
    
    text_lower = text.lower()
    matched = sum(1 for kw in keywords if kw in text_lower)
    match_ratio = matched / len(keywords) if keywords else 0.0
    
    # 모든 키워드가 매칭되면 추가 보너스
    if matched == len(keywords) and len(keywords) >= 2:
        return min(1.0, match_ratio * 1.5)  # 모든 키워드 매칭 시 50% 보너스
    
    return match_ratio

def enhance_search_scores(
    df: pd.DataFrame,
    query: str,
    faiss_scores: np.ndarray,
    faiss_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAISS 점수를 키워드 매칭으로 보정
    
    Returns:
        (enhanced_scores, indices) - 보정된 점수와 인덱스
    """
    keywords, query_region = split_query_into_keywords(query)
    
    enhanced_scores = []
    enhanced_indices = []
    
    for score, idx in zip(faiss_scores, faiss_indices):
        row = df.iloc[idx]
        
        # 기본 점수
        enhanced_score = float(score)
        
        # 제목, 내용, 지원대상, 지역 필드 가져오기
        title = str(row.get("제목", "") or "")
        content = str(row.get("지원내용", "") or "")
        target = str(row.get("지원대상", "") or "")
        region = str(row.get("지역", "") or "")
        
        # 키워드 매칭은 제목과 내용에서만 확인 (카테고리는 제외)
        all_text = (title + " " + content).lower()
        
        # 지역명 매칭 확인 (우선 처리)
        region_match = False
        if query_region:
            region_lower = region.lower()
            # 지역명이 정확히 포함되는지 확인
            region_match = (query_region in region_lower or 
                          region_lower in query_region or
                          any(qr in region_lower for qr in query_region.split()))
            
            if not region_match:
                # 지역 불일치 시 큰 감점
                enhanced_score *= 0.3
                enhanced_scores.append(enhanced_score)
                enhanced_indices.append(idx)
                continue  # 지역 불일치면 점수 낮게 유지
        
        # 키워드 매칭 점수 계산 (가중치 적용)
        keyword_score = 0.0
        if keywords:
            # 제목에 키워드가 있는지 확인 (가장 중요)
            title_match = calculate_keyword_match_score(keywords, title) * 0.6  # 제목 가중치 더 증가
            content_match = calculate_keyword_match_score(keywords, content) * 0.25
            target_match = calculate_keyword_match_score(keywords, target) * 0.1
            region_match_score = calculate_keyword_match_score(keywords, region) * 0.05
            
            keyword_score = title_match + content_match + target_match + region_match_score
            
            # 제목에 모든 키워드가 있는지 확인 (최고 보너스)
            title_lower = title.lower()
            all_in_title = all(kw in title_lower for kw in keywords)
            
            # 모든 키워드가 매칭되는지 확인 (제목+내용에서만)
            matched_keywords = sum(1 for kw in keywords if kw in all_text)
            match_ratio = matched_keywords / len(keywords) if keywords else 0.0
            
            # 제목에 키워드가 몇 개나 있는지 확인
            title_keywords_matched = sum(1 for kw in keywords if kw in title_lower)
            
            # 모든 키워드가 매칭되는 경우 (제목 또는 전체 텍스트)
            if all_in_title and len(keywords) >= 2:
                # 제목에 모든 키워드 있으면 최고 점수
                keyword_score = 1000.0  # 최고 점수로 설정 (매우 높은 값)
            elif match_ratio >= 1.0 and len(keywords) >= 2:
                # 모든 키워드가 매칭되면 매우 높은 점수
                keyword_score = 500.0  # 모든 키워드 매칭 시 높은 점수
            elif match_ratio < 1.0 and len(keywords) >= 2:
                # 일부 키워드만 매칭되면 점수 대폭 감소
                # 특히 제목에 키워드가 없으면 더 큰 감점
                if title_keywords_matched == 0:
                    keyword_score = keyword_score * 0.001  # 제목에 키워드 없으면 거의 0점
                else:
                    keyword_score = keyword_score * match_ratio * 0.01  # 일부만 매칭 시 매우 큰 감점
            
            # 제목에 모든 키워드가 있으면 최고 점수로 설정
            if all_in_title:
                enhanced_score = keyword_score  # 제목에 모든 키워드 있으면 키워드 점수만 사용
            elif match_ratio >= 1.0:
                # 모든 키워드가 매칭되면 키워드 점수 우선
                enhanced_score = keyword_score * 0.9 + enhanced_score * 0.1
            else:
                # 일부만 매칭되면 매우 낮은 점수 (FAISS 점수 무시)
                enhanced_score = keyword_score * 0.01  # 일부만 매칭 시 거의 0점
        else:
            # 키워드가 없으면 FAISS 점수만 사용
            enhanced_score = enhanced_score * 0.7
        
        # 지역 일치 보너스
        if query_region and region_match:
            enhanced_score += 0.3  # 지역 일치 보너스
        
        enhanced_scores.append(enhanced_score)
        enhanced_indices.append(idx)
    
    # 점수 순으로 정렬
    sorted_pairs = sorted(zip(enhanced_scores, enhanced_indices), key=lambda x: -x[0])
    enhanced_scores = np.array([s for s, _ in sorted_pairs])
    enhanced_indices = np.array([i for _, i in sorted_pairs])
    
    return enhanced_scores, enhanced_indices

