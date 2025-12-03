package com.policysearch

/**
 * 정책 검색 결과의 최종 점수 계산
 * 여러 요소를 종합하여 정확한 순위 결정
 */
class ScoreCalculator(synonymExpander: SynonymExpander) {
  
  /**
   * 종합 점수 계산
   * - 텍스트 매칭 점수 (0.0 ~ 1.0)
   * - 필터 일치 보너스
   * - 지역 일치 보너스
   * - 나이 일치 보너스
   */
  def calculateFinalScore(
    query: String,
    title: String,
    content: String,
    target: String,
    region: String,
    policyRegion: String,
    categories: Option[List[String]],
    policyCategories: Option[String],
    supports: Option[List[String]],
    policySupports: Option[String],
    age: Option[Int],
    ageRanges: Option[List[(Int, Int)]],
    baseScore: Double = 0.0 // FAISS 등에서 받은 기본 점수
  ): Double = {
    
    // 1. 동의어 확장된 쿼리로 텍스트 매칭 점수 계산
    val expandedQuery = synonymExpander.expandQuery(query)
    val textScore = TextMatcher.calculateWeightedScore(
      query,
      title,
      content,
      target,
      policyRegion,
      expandedQuery
    )
    
    // 2. 필터 일치 보너스
    var filterBonus = 0.0
    
    // 카테고리 일치 보너스
    if (categories.isDefined && policyCategories.isDefined) {
      val policyCatLower = policyCategories.get.toLowerCase
      val matched = categories.get.exists(cat => 
        policyCatLower.contains(cat.toLowerCase)
      )
      if (matched) filterBonus += 0.1
    }
    
    // 지원형태 일치 보너스
    if (supports.isDefined && policySupports.isDefined) {
      val policySupLower = policySupports.get.toLowerCase
      val matched = supports.get.exists(sup => 
        policySupLower.contains(sup.toLowerCase)
      )
      if (matched) filterBonus += 0.1
    }
    
    // 3. 지역 일치 보너스 (쿼리에 지역명이 포함된 경우 더 강하게 적용)
    var regionBonus = 0.0
    val (_, queryRegionOpt) = TextMatcher.splitQueryIntoKeywords(query)
    
    // 쿼리에 지역명이 포함되어 있으면 지역 매칭 보너스 강화
    if (queryRegionOpt.isDefined) {
      val queryRegion = queryRegionOpt.get
      if (TextMatcher.matchRegion(queryRegion, policyRegion)) {
        regionBonus += 0.25  // 쿼리 지역명 일치 시 큰 보너스
      }
    } else if (region.nonEmpty && region != "전국") {
      // 필터로 지정된 지역
      if (TextMatcher.matchRegion(region, policyRegion)) {
        regionBonus += 0.15
      }
    }
    
    // 4. 나이 일치 보너스
    var ageBonus = 0.0
    if (age.isDefined && ageRanges.isDefined) {
      if (TextMatcher.matchAge(age.get, ageRanges.get)) {
        ageBonus += 0.1
      }
    }
    
    // 5. 최종 점수 계산 (텍스트 점수 60%, 필터/지역/나이 보너스 40%)
    val finalScore = (textScore * 0.6) + 
                     (baseScore * 0.2) + 
                     filterBonus + 
                     regionBonus + 
                     ageBonus
    
    // 점수를 0.0 ~ 1.0 범위로 정규화
    math.min(1.0, math.max(0.0, finalScore))
  }
  
  /**
   * 정책 데이터로부터 점수 계산 (간편 버전)
   */
  def calculateScoreFromPolicy(
    query: String,
    policy: PolicyData,
    filters: SearchFilters
  ): Double = {
    calculateFinalScore(
      query = query,
      title = policy.title,
      content = policy.content,
      target = policy.target,
      region = filters.region.getOrElse(""),
      policyRegion = policy.region,
      categories = filters.categories,
      policyCategories = Some(policy.category),
      supports = filters.supports,
      policySupports = Some(policy.support),
      age = filters.age,
      ageRanges = policy.ageRanges,
      baseScore = 0.0
    )
  }
}

/**
 * 정책 데이터 모델
 */
case class PolicyData(
  index: Int,
  title: String,
  region: String,
  category: String,
  support: String,
  content: String,
  target: String,
  ageRanges: Option[List[(Int, Int)]]
)

/**
 * 검색 필터 모델
 */
case class SearchFilters(
  region: Option[String] = None,
  age: Option[Int] = None,
  categories: Option[List[String]] = None,
  supports: Option[List[String]] = None
)

