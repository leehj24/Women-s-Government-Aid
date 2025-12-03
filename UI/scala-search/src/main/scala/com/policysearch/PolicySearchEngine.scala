package com.policysearch

import org.json4s._
import org.json4s.native.JsonMethods._
import scala.io.Source
import scala.util.{Try, Success, Failure}
import java.io.File

/**
 * 정확한 정책 검색 엔진
 * Scala의 타입 안전성과 함수형 프로그래밍을 활용한 정교한 검색 알고리즘
 */
class PolicySearchEngine(synonymsPath: String, policiesPath: String) {
  
  private val synonymExpander = new SynonymExpander(synonymsPath)
  private val scoreCalculator = new ScoreCalculator(synonymExpander)
  private var policies: List[PolicyData] = List.empty
  
  /**
   * 정책 데이터 로드
   */
  def loadPolicies(): Unit = {
    policies = loadPoliciesFromFile(policiesPath)
    println(s"Loaded ${policies.length} policies")
  }
  
  private def loadPoliciesFromFile(path: String): List[PolicyData] = {
    Try {
      val file = new File(path)
      if (!file.exists()) {
        println(s"Warning: policies file not found at $path")
        return List.empty
      }
      
      val extension = file.getName.toLowerCase
      if (!extension.endsWith(".csv")) {
        println(s"Warning: Only CSV files are supported. Found: $extension")
        return List.empty
      }
      
      // CSV 파일 읽기
      val lines = Source.fromFile(file, "UTF-8").getLines().toList
      if (lines.isEmpty) {
        println("Warning: Empty file")
        return List.empty
      }
      
      val headers = parseCSVLine(lines.head)
      val titleIdx = headers.indexWhere(h => h == "제목" || h.toLowerCase.contains("title"))
      val regionIdx = headers.indexWhere(h => h == "지역" || h.toLowerCase.contains("region"))
      val categoryIdx = headers.indexWhere(h => 
        h.contains("카테고리") || h.toLowerCase.contains("category") || h.contains("분류")
      )
      val supportIdx = headers.indexWhere(h => 
        h.contains("지원형태") || h.toLowerCase.contains("support")
      )
      val contentIdx = headers.indexWhere(h => 
        h.contains("지원내용") || h.toLowerCase.contains("content")
      )
      val targetIdx = headers.indexWhere(h => 
        h.contains("지원대상") || h.toLowerCase.contains("target")
      )
      val ageIdx = headers.indexWhere(h => h.contains("age_eff_ranges"))
      
      if (titleIdx < 0 || regionIdx < 0) {
        println(s"Warning: Required columns not found. Title: $titleIdx, Region: $regionIdx")
        return List.empty
      }
      
      lines.tail.zipWithIndex.flatMap { case (line, idx) =>
        if (line.trim.isEmpty) None
        else {
          Try {
            val values = parseCSVLine(line)
            if (values.length <= math.max(titleIdx, regionIdx)) None
            else {
              val ageRanges = if (ageIdx >= 0 && ageIdx < values.length && values(ageIdx).nonEmpty) {
                parseAgeRanges(values(ageIdx))
              } else None
              
              Some(PolicyData(
                index = idx,
                title = safeGet(values, titleIdx, ""),
                region = safeGet(values, regionIdx, "전국"),
                category = safeGet(values, categoryIdx, ""),
                support = safeGet(values, supportIdx, ""),
                content = safeGet(values, contentIdx, ""),
                target = safeGet(values, targetIdx, ""),
                ageRanges = ageRanges
              ))
            }
          } match {
            case Success(opt) => opt
            case Failure(e) =>
              // 개별 행 오류는 무시하고 계속 진행
              None
          }
        }
      }
    } match {
      case Success(list) => 
        println(s"Successfully loaded ${list.length} policies")
        list
      case Failure(e) =>
        println(s"Error loading policies: ${e.getMessage}")
        e.printStackTrace()
        List.empty
    }
  }
  
  private def parseCSVLine(line: String): Array[String] = {
    // CSV 파싱 (쉼표로 분리, 따옴표 처리)
    val result = scala.collection.mutable.ArrayBuffer[String]()
    var current = new StringBuilder
    var inQuotes = false
    var i = 0
    
    while (i < line.length) {
      val c = line(i)
      c match {
        case '"' if !inQuotes =>
          inQuotes = true
        case '"' if inQuotes && i + 1 < line.length && line(i + 1) == '"' =>
          current.append('"')
          i += 1
        case '"' if inQuotes =>
          inQuotes = false
        case ',' if !inQuotes =>
          result += current.toString.trim
          current.clear()
        case _ =>
          current.append(c)
      }
      i += 1
    }
    result += current.toString.trim
    
    result.toArray
  }
  
  private def safeGet(arr: Array[String], idx: Int, default: String): String = {
    if (idx >= 0 && idx < arr.length && arr(idx).nonEmpty) arr(idx) else default
  }
  
  private def parseAgeRanges(ageStr: String): Option[List[(Int, Int)]] = {
    if (ageStr == null || ageStr.isEmpty) return None
    
    Try {
      implicit val formats: DefaultFormats.type = DefaultFormats
      val json = parse(ageStr)
      val ranges = json.extract[List[List[Int]]]
      Some(ranges.map { case List(min, max) => (min, max) })
    } match {
      case Success(ranges) => ranges
      case Failure(_) => None
    }
  }
  
  /**
   * 검색 실행
   */
  def search(
    query: String,
    filters: SearchFilters,
    topK: Int = 50
  ): List[SearchResult] = {
    
    if (policies.isEmpty) {
      loadPolicies()
    }
    
    if (policies.isEmpty) {
      return List.empty
    }
    
    val queryLower = query.toLowerCase.trim
    
    // 빈 쿼리인 경우 추천 모드로 전환
    if (queryLower.isEmpty) {
      return recommend(filters, topK)
    }
    
    // 각 정책에 대해 점수 계산
    val scored = policies.map { policy =>
      val score = scoreCalculator.calculateScoreFromPolicy(queryLower, policy, filters)
      SearchResult(policy, score)
    }
    
    // 필터링 및 정렬
    val (_, queryRegionOpt) = TextMatcher.splitQueryIntoKeywords(queryLower)
    
    val filtered = scored.filter { result =>
      var pass = true
      
      // 지역 필터 (쿼리에 지역명이 포함된 경우 우선 적용)
      if (queryRegionOpt.isDefined) {
        // 쿼리에 지역명이 있으면 해당 지역과 일치하는 것만 통과
        pass = pass && TextMatcher.matchRegion(queryRegionOpt.get, result.policy.region)
      } else if (filters.region.isDefined && filters.region.get != "전국") {
        // 필터로 지정된 지역
        pass = pass && TextMatcher.matchRegion(filters.region.get, result.policy.region)
      }
      
      // 나이 필터
      if (filters.age.isDefined && result.policy.ageRanges.isDefined) {
        pass = pass && TextMatcher.matchAge(filters.age.get, result.policy.ageRanges.get)
      }
      
      // 카테고리 필터
      if (filters.categories.isDefined && filters.categories.get.nonEmpty) {
        val policyCat = result.policy.category.toLowerCase
        pass = pass && filters.categories.get.exists(cat => 
          policyCat.contains(cat.toLowerCase)
        )
      }
      
      // 지원형태 필터
      if (filters.supports.isDefined && filters.supports.get.nonEmpty) {
        val policySup = result.policy.support.toLowerCase
        pass = pass && filters.supports.get.exists(sup => 
          policySup.contains(sup.toLowerCase)
        )
      }
      
      pass && result.score > 0.0 // 최소 점수 이상만
    }
    
    // 점수 순으로 정렬하고 상위 K개 반환
    filtered.sortBy(-_.score).take(topK)
  }
  
  /**
   * 추천 (필터만 사용, 쿼리 없음)
   */
  def recommend(filters: SearchFilters, topK: Int = 50): List[SearchResult] = {
    if (policies.isEmpty) {
      loadPolicies()
    }
    
    // 필터만 적용하여 추천
    val filtered = policies.map { policy =>
      var score = 0.5 // 기본 점수
      
      // 필터 일치 보너스
      if (filters.region.isDefined && filters.region.get != "전국") {
        if (TextMatcher.matchRegion(filters.region.get, policy.region)) {
          score += 0.2
        }
      }
      
      if (filters.age.isDefined && policy.ageRanges.isDefined) {
        if (TextMatcher.matchAge(filters.age.get, policy.ageRanges.get)) {
          score += 0.2
        }
      }
      
      if (filters.categories.isDefined) {
        val policyCat = policy.category.toLowerCase
        if (filters.categories.get.exists(cat => policyCat.contains(cat.toLowerCase))) {
          score += 0.1
        }
      }
      
      SearchResult(policy, math.min(1.0, score))
    }.filter(_.score > 0.3)
    
    filtered.sortBy(-_.score).take(topK)
  }
}

/**
 * 검색 결과 모델
 */
case class SearchResult(policy: PolicyData, score: Double)

