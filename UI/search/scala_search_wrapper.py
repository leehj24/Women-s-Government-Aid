# scala_search_wrapper.py
# Scala 검색 엔진을 Python에서 호출하는 래퍼

import json
import subprocess
import os
import pathlib
import logging
from typing import List, Optional, Dict, Any
import pandas as pd

log = logging.getLogger(__name__)

class ScalaSearchWrapper:
    """Scala 검색 엔진을 Python에서 호출하는 래퍼 클래스"""
    
    def __init__(self, scala_jar_path: Optional[str] = None, 
                 synonyms_path: Optional[str] = None,
                 policies_path: Optional[str] = None):
        """
        Args:
            scala_jar_path: Scala JAR 파일 경로 (없으면 자동 탐색)
            synonyms_path: 동의어 JSON 파일 경로
            policies_path: 정책 데이터 파일 경로
        """
        self.base_dir = pathlib.Path(__file__).resolve().parent.parent
        
        # 경로 자동 탐색
        if scala_jar_path is None:
            jar_candidates = [
                self.base_dir / "scala-search" / "target" / "scala-2.13" / "policy-search-engine.jar",
                self.base_dir / "scala-search" / "target" / "policy-search-engine.jar",
            ]
            self.scala_jar_path = next((p for p in jar_candidates if p.exists()), None)
        else:
            self.scala_jar_path = pathlib.Path(scala_jar_path)
        
        if synonyms_path is None:
            self.synonyms_path = self.base_dir / "synonyms.json"
        else:
            self.synonyms_path = pathlib.Path(synonyms_path)
        
        if policies_path is None:
            # 정책 파일 자동 탐색 (loader.py와 동일한 로직)
            from .loader import _resolve_file
            try:
                resolved_path = _resolve_file()
                # CSV만 지원하므로 CSV로 변환 필요시 처리
                if resolved_path.suffix.lower() == ".csv":
                    self.policies_path = resolved_path
                else:
                    # Excel 파일인 경우 CSV로 변환된 파일 찾기
                    csv_candidates = [
                        self.base_dir / "policy_summary_langchain_streaming.csv",
                        resolved_path.parent / "policy_summary_langchain_streaming.csv",
                    ]
                    self.policies_path = next((p for p in csv_candidates if p.exists()), None)
                    if self.policies_path is None:
                        log.warning(f"Excel file found but CSV required: {resolved_path}")
            except Exception as e:
                log.warning(f"Could not resolve policy file: {e}")
                # 폴백: 직접 탐색
                policy_candidates = [
                    self.base_dir / "policy_summary_langchain_streaming.csv",
                    self.base_dir / "정책큐레이션_통합데이터_v1.0.xlsx",
                ]
                self.policies_path = next((p for p in policy_candidates if p.exists()), None)
        else:
            self.policies_path = pathlib.Path(policies_path)
        
        self._check_paths()
    
    def _check_paths(self):
        """필수 파일 경로 확인"""
        if self.scala_jar_path is None or not self.scala_jar_path.exists():
            log.warning(f"Scala JAR not found at {self.scala_jar_path}. "
                       "Please build the Scala project first: cd scala-search && sbt assembly")
            self.available = False
            return
        
        if not self.synonyms_path.exists():
            log.warning(f"Synonyms file not found at {self.synonyms_path}")
            self.available = False
            return
        
        if self.policies_path is None or not self.policies_path.exists():
            log.warning(f"Policies file not found at {self.policies_path}")
            self.available = False
            return
        
        self.available = True
    
    def _call_scala_engine(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scala 엔진 호출"""
        if not self.available:
            raise RuntimeError("Scala search engine is not available. Check paths and build status.")
        
        try:
            # JSON 요청 생성
            json_input = json.dumps(request, ensure_ascii=False)
            
            # Scala 프로그램 실행
            cmd = [
                "java", "-jar", str(self.scala_jar_path),
                str(self.synonyms_path),
                str(self.policies_path)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=str(self.base_dir)
            )
            
            stdout, stderr = process.communicate(input=json_input, timeout=60)
            
            if process.returncode != 0:
                log.error(f"Scala engine error: {stderr}")
                raise RuntimeError(f"Scala engine failed: {stderr}")
            
            # 결과 파싱
            results = json.loads(stdout)
            return results
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("Scala engine timeout")
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse Scala engine output: {e}")
            raise RuntimeError(f"Invalid response from Scala engine: {e}")
        except Exception as e:
            log.error(f"Error calling Scala engine: {e}")
            raise
    
    def search(self, 
               query: str,
               region: Optional[str] = None,
               age: Optional[int] = None,
               categories: Optional[List[str]] = None,
               supports: Optional[List[str]] = None,
               top_k: int = 50) -> pd.DataFrame:
        """
        검색 실행
        
        Returns:
            DataFrame with columns: index, title, region, category, support, score
        """
        request = {
            "type": "search",
            "query": query,
            "region": region,
            "age": age,
            "categories": categories,
            "supports": supports,
            "topK": top_k
        }
        
        results = self._call_scala_engine(request)
        
        if not results:
            return pd.DataFrame(columns=["index", "title", "region", "category", "support", "score"])
        
        return pd.DataFrame(results)
    
    def recommend(self,
                  region: Optional[str] = None,
                  age: Optional[int] = None,
                  categories: Optional[List[str]] = None,
                  supports: Optional[List[str]] = None,
                  top_k: int = 50) -> pd.DataFrame:
        """
        추천 (필터만 사용)
        
        Returns:
            DataFrame with columns: index, title, region, category, support, score
        """
        request = {
            "type": "recommend",
            "query": "",
            "region": region,
            "age": age,
            "categories": categories,
            "supports": supports,
            "topK": top_k
        }
        
        results = self._call_scala_engine(request)
        
        if not results:
            return pd.DataFrame(columns=["index", "title", "region", "category", "support", "score"])
        
        return pd.DataFrame(results)


# 전역 인스턴스 (싱글톤 패턴)
_scala_wrapper: Optional[ScalaSearchWrapper] = None

def get_scala_wrapper() -> Optional[ScalaSearchWrapper]:
    """Scala 래퍼 인스턴스 가져오기 (지연 초기화)"""
    global _scala_wrapper
    if _scala_wrapper is None:
        try:
            _scala_wrapper = ScalaSearchWrapper()
        except Exception as e:
            log.warning(f"Failed to initialize Scala wrapper: {e}")
            return None
    return _scala_wrapper if _scala_wrapper.available else None

