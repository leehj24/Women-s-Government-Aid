# 검색 개선 사항 상세 설명

## 📋 개요

기존 Python FAISS 엔진과 새로운 Scala 검색 엔진의 차이점을 실제 사용 시나리오별로 설명합니다.

---

## 🔍 시나리오 1: 맞춤정보 검색 (쿼리 없음)

### 사용자 동작
1. 지역 선택: "서울"
2. 생년월일 입력: "1990-01-15" (만 34세)
3. 카테고리 선택: "의료비", "건강검진"
4. "검색" 버튼 클릭

### 기존 Python 엔진 방식

**코드 경로:** `UI/search/engine.py` → `recommend()` 메서드

```python
# 1단계: 필터만 적용 (카테고리, 지역, 나이)
mask = self._stage1_mask(...)  # True/False 배열
tmp = self.df[mask].copy()

# 2단계: 신청기간 마감일 기준 정렬
tmp["_end"] = tmp["신청기간"].apply(_last_date_from_period)
tmp = tmp.sort_values(by=["_end","제목"], ascending=[True,True])

# 결과: 마감 임박 순서로 정렬된 정책 리스트
```

**문제점:**
- 단순 필터링만 수행
- 마감일 기준 정렬만 있음
- 사용자 맞춤도 점수 없음
- 카테고리 일치 여부만 확인 (부분 일치)

### 새로운 Scala 엔진 방식

**코드 경로:** `UI/scala-search/.../PolicySearchEngine.scala` → `recommend()` 메서드

```scala
// 각 정책에 대해 맞춤도 점수 계산
val scored = policies.map { policy =>
  var score = 0.5  // 기본 점수
  
  // 지역 일치 보너스 (+0.2)
  if (TextMatcher.matchRegion("서울", policy.region)) {
    score += 0.2
  }
  
  // 나이 일치 보너스 (+0.2)
  if (TextMatcher.matchAge(34, policy.ageRanges)) {
    score += 0.2
  }
  
  // 카테고리 일치 보너스 (+0.1)
  if (policy.category.contains("의료비") || 
      policy.category.contains("건강검진")) {
    score += 0.1
  }
  
  SearchResult(policy, score)
}

// 점수 높은 순으로 정렬
filtered.sortBy(-_.score).take(50)
```

**개선점:**
- ✅ **맞춤도 점수 계산**: 지역/나이/카테고리 일치도에 따라 점수 부여
- ✅ **정확한 정렬**: 단순 마감일이 아닌 사용자 맞춤도 순서
- ✅ **더 관련성 높은 결과**: 점수가 높은 정책이 상위에 표시

---

## 🔍 시나리오 2: 맞춤정보 + 추가 검색어 입력

### 사용자 동작
1. 위의 맞춤정보 설정 후
2. 검색어 입력: "임산부"
3. "검색" 버튼 클릭

### 기존 Python 엔진 방식

**코드 경로:** `UI/search/engine.py` → `search()` 메서드

```python
# 1단계: FAISS 벡터 검색
D, I = self.index.query("임산부", topk=None)
# → 모든 정책에 대한 유사도 점수 (0.0 ~ 1.0)

# 2단계: 필터 적용 (교집합)
mask = self._stage1_mask(..., kw_text="임산부")
for rank, idx in enumerate(order):
    sc = float(scores[rank])
    if sc < threshold: continue  # 임계값 미만 제외
    if not mask[idx]: continue    # 필터 불일치 제외
    rows_yes.append((sc, idx))

# 3단계: 점수 순 정렬
rows = sorted(rows_yes, key=lambda x: -x[0])[:200]
```

**문제점:**
- FAISS 벡터 유사도만 사용 (의미적 유사도)
- 필터는 단순 True/False (보너스 없음)
- 제목/내용 가중치 없음 (모든 필드 동일 취급)
- 동의어 확장 없음 ("임산부"만 검색, "임신부", "산모" 등 미포함)

### 새로운 Scala 엔진 방식

**코드 경로:** `UI/scala-search/.../PolicySearchEngine.scala` → `search()` 메서드

```scala
// 1단계: 동의어 확장
val expandedQuery = synonymExpander.expandQuery("임산부")
// → Set("임산부", "임신부", "산모", "임부", "예비맘", "임신여성", ...)

// 2단계: 각 정책에 대해 가중치 기반 점수 계산
val scored = policies.map { policy =>
  // 텍스트 매칭 점수 (가중치 적용)
  val titleScore = TextMatcher.calculateMultiKeywordScore(
    expandedQuery, policy.title
  ) * 0.4  // 제목 가중치 40%
  
  val contentScore = TextMatcher.calculateMultiKeywordScore(
    expandedQuery, policy.content
  ) * 0.3  // 내용 가중치 30%
  
  val targetScore = TextMatcher.calculateMultiKeywordScore(
    expandedQuery, policy.target
  ) * 0.2  // 지원대상 가중치 20%
  
  val regionScore = TextMatcher.calculateMultiKeywordScore(
    expandedQuery, policy.region
  ) * 0.1  // 지역 가중치 10%
  
  val textScore = titleScore + contentScore + targetScore + regionScore
  
  // 필터 보너스
  var filterBonus = 0.0
  if (matchRegion("서울", policy.region)) filterBonus += 0.15
  if (matchAge(34, policy.ageRanges)) filterBonus += 0.1
  if (policy.category.contains("의료비")) filterBonus += 0.1
  
  // 최종 점수
  val finalScore = (textScore * 0.6) + filterBonus
  SearchResult(policy, finalScore)
}

// 3단계: 필터링 및 정렬
filtered.sortBy(-_.score).take(50)
```

**개선점:**
- ✅ **동의어 확장**: "임산부" 검색 시 "임신부", "산모" 등도 매칭
- ✅ **필드별 가중치**: 제목(40%) > 내용(30%) > 지원대상(20%) > 지역(10%)
- ✅ **정확한 매칭**: 정확 일치(1.0) > 시작 일치(0.8) > 포함(0.6) > 단어 매칭(0.4)
- ✅ **필터 보너스**: 지역/나이/카테고리 일치 시 추가 점수
- ✅ **종합 점수**: 텍스트 매칭(60%) + 필터 보너스(40%)

---

## 🔍 시나리오 3: 전체 검색 (조건검색 + 검색어)

### 사용자 동작
1. 전체검색 탭 선택
2. 조건검색 열기
3. 지역: "부산", 지원형태: "바우처" 선택
4. 검색어 입력: "육아 지원"
5. "검색" 버튼 클릭

### 기존 Python 엔진 방식

```python
# 1단계: FAISS 벡터 검색
D, I = self.index.query("육아 지원", topk=None)

# 2단계: 필터 적용
mask = self._stage1_mask(
    categories=None,
    supports=["바우처"],
    region="부산",
    dob=None,
    kw_text="육아 지원"
)

# 3단계: 필터 통과한 것만 점수 순 정렬
rows_yes = []
for rank, idx in enumerate(order):
    if scores[rank] < threshold: continue
    if not mask[idx]: continue  # 필터 불일치 제외
    rows_yes.append((scores[rank], idx))
```

**문제점:**
- "육아"와 "지원"을 별도로 처리하지 않음
- "육아" 동의어("보육", "양육" 등) 미확장
- 지원형태 필터는 단순 포함 여부만 확인
- 지역 필터도 단순 포함 여부만 확인

### 새로운 Scala 엔진 방식

```scala
// 1단계: 검색어 동의어 확장
val expandedQuery = synonymExpander.expandQuery("육아 지원")
// → Set("육아", "보육", "양육", "자녀돌봄", "돌봄서비스", 
//       "지원", "지원금", "지원비", ...)

// 2단계: 각 정책 점수 계산
val scored = policies.map { policy =>
  // 텍스트 매칭 (가중치 적용)
  val textScore = calculateWeightedScore(
    "육아 지원",
    policy.title,      // 40% 가중치
    policy.content,    // 30% 가중치
    policy.target,     // 20% 가중치
    policy.region      // 10% 가중치
  )
  
  // 필터 보너스
  var bonus = 0.0
  
  // 지역 일치 보너스
  if (matchRegion("부산", policy.region)) bonus += 0.15
  
  // 지원형태 일치 보너스
  if (policy.support.contains("바우처")) bonus += 0.1
  
  // 최종 점수
  (textScore * 0.6) + bonus
}

// 3단계: 필터링
val filtered = scored.filter { result =>
  // 지역 필터
  matchRegion("부산", result.policy.region) &&
  // 지원형태 필터
  result.policy.support.contains("바우처") &&
  // 최소 점수 이상
  result.score > 0.0
}

// 4단계: 점수 순 정렬
filtered.sortBy(-_.score).take(50)
```

**개선점:**
- ✅ **다중 키워드 처리**: "육아 지원" → "육아" + "지원" 각각 동의어 확장
- ✅ **동의어 확장**: "육아" → "보육", "양육", "자녀돌봄" 등
- ✅ **정확한 지역 매칭**: "부산" 검색 시 "부산광역시", "부산시" 등도 매칭
- ✅ **지원형태 정확 매칭**: "바우처" 포함 여부를 정확히 확인
- ✅ **종합 점수**: 텍스트 매칭 + 필터 보너스를 종합한 최종 점수

---

## 📊 점수 계산 방식 비교

### 기존 Python 엔진
```
최종 점수 = FAISS 벡터 유사도 (0.0 ~ 1.0)
필터 = True/False (점수 영향 없음)
```

### 새로운 Scala 엔진
```
최종 점수 = 
  텍스트 매칭 점수 (60%)
    = 제목 매칭 (40% 가중치)
    + 내용 매칭 (30% 가중치)
    + 지원대상 매칭 (20% 가중치)
    + 지역 매칭 (10% 가중치)
  
  + 필터 보너스 (40%)
    + 지역 일치 보너스 (최대 0.15)
    + 나이 일치 보너스 (최대 0.1)
    + 카테고리 일치 보너스 (최대 0.1)
    + 지원형태 일치 보너스 (최대 0.1)
```

---

## 🎯 실제 차이 예시

### 예시: "임산부 의료비" 검색

**기존 방식:**
- "임산부 의료비" 벡터와 유사한 정책만 찾음
- "임신부 의료비 지원" 정책은 낮은 점수 (의미적 유사도만)
- "산모 건강검진" 정책은 더 낮은 점수

**새로운 방식:**
- "임산부" → "임신부", "산모", "임부" 등 동의어 확장
- "의료비" → "진료비", "치료비", "병원비" 등 동의어 확장
- 제목에 "임산부"가 있으면 높은 점수 (40% 가중치)
- 내용에 "의료비 지원"이 있으면 중간 점수 (30% 가중치)
- 지역/나이 일치 시 추가 보너스

**결과:**
- 더 관련성 높은 정책이 상위에 표시
- 사용자가 찾는 정책을 더 정확하게 찾음

---

## 🔧 코드 흐름도

### 전체 검색 흐름

```
사용자 입력
    ↓
[app.js] fetchResults() → POST /search
    ↓
[app.py] search() → find_policies()
    ↓
[policy_search.py] find_policies()
    ├─ Scala 엔진 사용 가능?
    │   ├─ YES → [scala_search_wrapper.py] search()
    │   │         ↓
    │   │         [Scala] PolicySearchEngine.search()
    │   │         ├─ 동의어 확장
    │   │         ├─ 가중치 점수 계산
    │   │         ├─ 필터 보너스 적용
    │   │         └─ 점수 순 정렬
    │   │
    │   └─ NO → [engine.py] SearchEngine.search()
    │             ├─ FAISS 벡터 검색
    │             ├─ 필터 적용 (True/False)
    │             └─ 점수 순 정렬
    ↓
결과 반환 (DataFrame)
    ↓
[app.py] JSON 변환
    ↓
[app.js] 결과 표시
```

---

## ✅ 요약

| 항목 | 기존 Python 엔진 | 새로운 Scala 엔진 |
|------|-----------------|------------------|
| **검색 방식** | FAISS 벡터 유사도 | 텍스트 매칭 + 가중치 |
| **동의어 확장** | ❌ 없음 | ✅ synonyms.json 활용 |
| **필드 가중치** | ❌ 없음 (동일 취급) | ✅ 제목(40%) > 내용(30%) > ... |
| **필터 처리** | True/False만 | 점수 보너스 추가 |
| **매칭 정확도** | 의미적 유사도만 | 정확 일치 > 부분 일치 > 단어 매칭 |
| **맞춤도 계산** | ❌ 없음 | ✅ 지역/나이/카테고리 일치도 반영 |
| **정렬 방식** | 벡터 유사도 순 | 종합 점수 순 |

**핵심 개선점:**
1. **더 정확한 검색**: 동의어 확장으로 관련 정책을 더 많이 찾음
2. **더 관련성 높은 결과**: 제목/내용 가중치로 중요한 정보가 상위에 표시
3. **더 맞춤된 추천**: 필터 일치도에 따라 점수 보너스 부여


