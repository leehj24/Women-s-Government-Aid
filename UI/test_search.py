# test_search.py
# 검색 기능 테스트 스크립트

import sys
sys.path.insert(0, '.')

from search.policy_search import find_policies
import pandas as pd

# 테스트 쿼리
query = "대구광역시 임산부 체육시설"
print(f"검색 쿼리: {query}\n")

# 검색 실행
df = find_policies(
    input=query,
    region="",
    dob="",
    categories=[],
    supports=[],
    out="dataframe",
    use_scala=False  # Python 엔진 사용
)

print(f"검색 결과: {len(df)}개\n")

# 상위 10개 결과 출력
if len(df) > 0:
    print("=== 상위 10개 결과 ===\n")
    for idx, row in df.head(10).iterrows():
        title = row.get("제목", "")
        region = row.get("지역", "")
        score = row.get("score", 0.0)
        
        # 키워드 매칭 확인
        title_lower = str(title).lower()
        keywords = ["대구", "임산부", "체육시설"]
        matched = [kw for kw in keywords if kw in title_lower]
        
        print(f"[{idx+1}] 점수: {score:.4f}")
        print(f"     제목: {title}")
        print(f"     지역: {region}")
        print(f"     매칭 키워드: {matched}")
        print()
else:
    print("검색 결과가 없습니다.")

