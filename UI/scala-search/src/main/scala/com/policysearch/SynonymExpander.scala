package com.policysearch

import org.json4s._
import org.json4s.native.JsonMethods._
import scala.io.Source
import scala.util.{Try, Success, Failure}
import java.io.File

/**
 * 동의어 확장 및 검색어 강화
 * synonyms.json 파일을 로드하여 검색어를 확장
 */
class SynonymExpander(synonymsPath: String) {
  private val synonymMap: Map[String, Set[String]] = loadSynonyms()
  
  private def loadSynonyms(): Map[String, Set[String]] = {
    Try {
      val file = new File(synonymsPath)
      if (!file.exists()) {
        println(s"Warning: synonyms file not found at $synonymsPath")
        return Map.empty[String, Set[String]]
      }
      
      val content = Source.fromFile(file, "UTF-8").mkString
      implicit val formats: DefaultFormats.type = DefaultFormats
      val json = parse(content)
      
      json.extract[Map[String, List[String]]].map { case (key, values) =>
        key -> (values.toSet + key) // 원본 키워드도 포함
      }
    } match {
      case Success(map) => map
      case Failure(e) =>
        println(s"Error loading synonyms: ${e.getMessage}")
        Map.empty[String, Set[String]]
    }
  }
  
  /**
   * 검색어를 동의어로 확장
   */
  def expandQuery(query: String): Set[String] = {
    val words = query.split("\\s+").filter(_.nonEmpty)
    val expanded = words.flatMap { word =>
      synonymMap.getOrElse(word, Set(word)) + word
    }.toSet
    
    // 원본 쿼리도 포함
    expanded + query
  }
  
  /**
   * 특정 키워드의 동의어 반환
   */
  def getSynonyms(keyword: String): Set[String] = {
    synonymMap.getOrElse(keyword, Set(keyword))
  }
  
  /**
   * 모든 동의어 맵 반환
   */
  def getAllSynonyms(): Map[String, Set[String]] = synonymMap
}

