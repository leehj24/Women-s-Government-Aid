package com.policysearch

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization.write

/**
 * Scala 검색 엔진의 메인 진입점
 * JSON 형식으로 검색 요청을 받아 결과를 반환
 */
object Main {
  
  implicit val formats: DefaultFormats.type = DefaultFormats
  
  def main(args: Array[String]): Unit = {
    try {
      if (args.length < 2) {
        System.err.println("Usage: Main <synonyms_path> <policies_path>")
        System.exit(1)
      }
      
      val synonymsPath = args(0)
      val policiesPath = args(1)
      
      val engine = new PolicySearchEngine(synonymsPath, policiesPath)
      engine.loadPolicies()
      
      // JSON 입력 읽기
      val input = scala.io.Source.stdin.mkString
      if (input.trim.isEmpty) {
        System.err.println("Error: Empty input")
        System.exit(1)
      }
      
      val json = parse(input)
      
      val query = (json \ "query").extractOrElse[String]("")
      val region = (json \ "region").extractOpt[String]
      val age = (json \ "age").extractOpt[Int]
      val categories = (json \ "categories").extractOpt[List[String]]
      val supports = (json \ "supports").extractOpt[List[String]]
      val topK = (json \ "topK").extractOrElse[Int](50)
      val searchType = (json \ "type").extractOrElse[String]("search")
      
      val filters = SearchFilters(region, age, categories, supports)
      
      val results = searchType match {
        case "recommend" => engine.recommend(filters, topK)
        case _ => engine.search(query, filters, topK)
      }
      
      // 결과를 JSON으로 변환
      val output = results.map { result =>
        Map(
          "index" -> result.policy.index,
          "title" -> result.policy.title,
          "region" -> result.policy.region,
          "category" -> result.policy.category,
          "support" -> result.policy.support,
          "score" -> result.score
        )
      }
      
      println(write(output))
      
    } catch {
      case e: Exception =>
        System.err.println(s"Error: ${e.getMessage}")
        e.printStackTrace()
        System.exit(1)
    }
  }
}

