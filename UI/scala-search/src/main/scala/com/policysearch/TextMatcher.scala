package com.policysearch

import scala.util.matching.Regex

/**
 * 정확한 텍스트 매칭을 위한 유틸리티
 * 한국어 특성을 고려한 매칭 알고리즘
 */
object TextMatcher {
  
  /**
   * 텍스트 정규화 (공백, 특수문자 처리)
   */
  def normalize(text: String): String = {
    if (text == null) return ""
    text
      .replaceAll("\\r\\n?", " ")
      .replaceAll("\\t", " ")
      .replaceAll("\\s+", " ")
      .trim
      .toLowerCase
  }
  
  /**
   * 부분 문자열 매칭 점수 계산
   * - 정확한 일치: 1.0
   * - 시작 부분 일치: 0.8
   * - 중간 포함: 0.6
   * - 단어 단위 일치: 0.4
   */
  def calculateMatchScore(query: String, text: String): Double = {
    val normQuery = normalize(query)
    val normText = normalize(text)
    
    if (normText.isEmpty || normQuery.isEmpty) return 0.0
    
    // 정확한 일치
    if (normText == normQuery) return 1.0
    
    // 시작 부분 일치
    if (normText.startsWith(normQuery)) return 0.8
    
    // 끝 부분 일치
    if (normText.endsWith(normQuery)) return 0.75
    
    // 완전 포함
    if (normText.contains(normQuery)) return 0.6
    
    // 단어 단위 매칭
    val queryWords = normQuery.split("\\s+").filter(_.nonEmpty)
    val textWords = normText.split("\\s+").filter(_.nonEmpty)
    
    if (queryWords.isEmpty) return 0.0
    
    val matchedWords = queryWords.count(qw => textWords.exists(tw => tw.contains(qw) || qw.contains(tw)))
    val wordMatchRatio = matchedWords.toDouble / queryWords.length
    
    wordMatchRatio * 0.4
  }
  
  /**
   * 다중 키워드 매칭 점수 계산
   * 모든 키워드가 매칭되면 보너스 점수 부여
   * 핵심 키워드(체육시설, 의료비 등)가 제목에 있으면 더 높은 점수
   */
  def calculateMultiKeywordScore(keywords: Set[String], text: String): Double = {
    if (keywords.isEmpty) return 0.0
    
    val normText = normalize(text)
    val scores = keywords.map { kw =>
      val score = calculateMatchScore(kw, normText)
      // 핵심 키워드가 제목에 정확히 매칭되면 추가 보너스
      val normKw = normalize(kw)
      if (normText.startsWith(normKw) || normText.contains(s" $normKw ") || normText.contains(s"$normKw ")) {
        score * 1.3  // 핵심 키워드 매칭 시 30% 보너스
      } else {
        score
      }
    }
    
    val avgScore = scores.sum / scores.size
    val matchedCount = scores.count(_ > 0.3)
    val allMatched = matchedCount == keywords.size
    
    // 모든 키워드가 매칭되면 큰 보너스
    if (allMatched) {
      avgScore * 1.5  // 모든 키워드 매칭 시 50% 보너스
    } else if (matchedCount >= keywords.size * 0.7) {
      avgScore * 1.2  // 70% 이상 매칭 시 20% 보너스
    } else {
      avgScore
    }
  }
  
  /**
   * 쿼리를 키워드로 분리 (지역명, 일반 키워드 구분)
   */
  def splitQueryIntoKeywords(query: String): (Set[String], Option[String]) = {
    val queryLower = query.toLowerCase.trim
    
    // 지역명 패턴 (광역시, 특별시, 도 등 포함)
    val regionPatterns = List(
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
    )
    
    // 쿼리에서 지역명 찾기 (긴 것부터 매칭)
    val sortedPatterns = regionPatterns.sortBy(-_.length)
    var foundRegion: Option[String] = None
    var remainingQuery = queryLower
    
    sortedPatterns.find { pattern =>
      if (remainingQuery.contains(pattern)) {
        foundRegion = Some(pattern)
        // 지역명 제거
        remainingQuery = remainingQuery.replace(pattern, " ").trim
        true
      } else {
        false
      }
    }
    
    // 남은 키워드 추출 (1글자 제외)
    val keywords = remainingQuery.split("\\s+")
      .filter(_.nonEmpty)
      .filter(w => w.length > 1)
      .toSet
    
    (keywords, foundRegion)
  }
  
  /**
   * 가중치 기반 필드별 매칭 점수 계산 (개선 버전)
   * 다중 키워드를 개별적으로 매칭하고 모든 키워드 매칭 시 보너스
   */
  def calculateWeightedScore(
    query: String,
    title: String,
    content: String,
    target: String,
    region: String,
    expandedKeywords: Set[String]
  ): Double = {
    val (queryKeywords, queryRegion) = splitQueryIntoKeywords(query)
    
    // 동의어 확장된 키워드에서 지역명 제외
    val expandedNonRegion = expandedKeywords.filter { kw =>
      !queryRegion.exists(reg => kw.contains(reg) || reg.contains(kw))
    }
    
    val allKeywords = queryKeywords ++ expandedNonRegion
    
    // 각 필드별로 키워드 매칭 점수 계산
    val titleScore = calculateMultiKeywordScore(allKeywords, title) * 0.4
    val contentScore = calculateMultiKeywordScore(allKeywords, content) * 0.3
    val targetScore = calculateMultiKeywordScore(allKeywords, target) * 0.2
    
    // 지역 필드는 지역명 매칭에 더 높은 가중치
    val regionScore = if (queryRegion.isDefined) {
      val queryReg = queryRegion.get
      if (matchRegion(queryReg, region)) {
        0.3  // 쿼리 지역명이 정책 지역과 일치하면 높은 점수
      } else {
        calculateMultiKeywordScore(allKeywords, region) * 0.1
      }
    } else {
      calculateMultiKeywordScore(allKeywords, region) * 0.1
    }
    
    var baseScore = titleScore + contentScore + targetScore + regionScore
    
    // 모든 키워드가 매칭되는 경우 큰 보너스
    val normTitle = normalize(title)
    val normContent = normalize(content)
    val normTarget = normalize(target)
    val normRegion = normalize(region)
    
    val allText = s"$normTitle $normContent $normTarget $normRegion"
    
    val matchedKeywords = allKeywords.count { kw =>
      val normKw = normalize(kw)
      allText.contains(normKw)
    }
    
    val keywordMatchRatio = if (allKeywords.nonEmpty) {
      matchedKeywords.toDouble / allKeywords.size
    } else 0.0
    
    // 모든 키워드가 매칭되면 보너스 (최대 0.4)
    if (keywordMatchRatio >= 0.9 && allKeywords.size >= 2) {
      baseScore += 0.4  // 거의 모든 키워드 매칭 시 큰 보너스
    } else if (keywordMatchRatio >= 0.8) {
      baseScore += 0.3
    } else if (keywordMatchRatio >= 0.6) {
      baseScore += 0.15
    }
    
    // 쿼리 지역명이 정책 지역과 일치하면 추가 보너스
    if (queryRegion.isDefined) {
      val queryReg = queryRegion.get
      if (matchRegion(queryReg, region)) {
        baseScore += 0.25  // 지역 일치 보너스 강화
      } else {
        // 지역 불일치 시 감점
        baseScore *= 0.5
      }
    }
    
    math.min(1.0, baseScore)
  }
  
  /**
   * 지역명 매칭 (시/도, 시/군/구 레벨)
   */
  def matchRegion(queryRegion: String, policyRegion: String): Boolean = {
    val normQuery = normalize(queryRegion)
    val normPolicy = normalize(policyRegion)
    
    if (normQuery == "전국" || normQuery.isEmpty) return true
    if (normPolicy == "전국") return true
    
    normPolicy.contains(normQuery) || normQuery.contains(normPolicy)
  }
  
  /**
   * 나이 범위 매칭
   */
  def matchAge(age: Int, ageRanges: List[(Int, Int)]): Boolean = {
    if (ageRanges.isEmpty) return true
    ageRanges.exists { case (min, max) => age >= min && age <= max }
  }
}

