# Scala 정책 검색 엔진

더 정확한 정책 검색을 위한 Scala 기반 검색 엔진입니다.

## 특징

- **동의어 확장**: synonyms.json을 활용한 검색어 확장
- **가중치 기반 점수 계산**: 제목, 내용, 지원대상, 지역 등 필드별 가중치 적용
- **정확한 매칭**: 한국어 특성을 고려한 텍스트 매칭 알고리즘
- **타입 안전성**: Scala의 강력한 타입 시스템으로 오류 방지

## 빌드 방법

### Windows
```bash
cd UI/scala-search
build.bat
```

### Linux/Mac
```bash
cd UI/scala-search
chmod +x build.sh
./build.sh
```

또는 직접 sbt 사용:
```bash
sbt assembly
```

## 사용 방법

Python에서 자동으로 호출됩니다. 수동으로 테스트하려면:

```bash
java -jar target/scala-2.13/policy-search-engine.jar synonyms.json policies.csv < input.json
```

입력 JSON 형식:
```json
{
  "type": "search",
  "query": "임산부 의료비",
  "region": "서울",
  "age": 30,
  "categories": ["의료비"],
  "supports": [],
  "topK": 50
}
```

## 의존성

- Scala 2.13.12
- sbt 1.9.7
- json4s-native 4.0.6

