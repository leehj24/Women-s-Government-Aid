# Scala 검색 엔진 설정 가이드

## 개요

더 정확한 정책 검색을 위해 Scala 기반 검색 엔진이 추가되었습니다. 
이 엔진은 동의어 확장, 가중치 기반 점수 계산, 정확한 텍스트 매칭을 제공합니다.

## 사전 요구사항

1. **Java JDK 8 이상** 설치 필요
   - 확인: `java -version`
   - 다운로드: https://adoptium.net/

2. **Scala 및 sbt 설치**
   - sbt만 설치하면 Scala가 자동으로 포함됩니다
   - Windows: https://www.scala-sbt.org/download.html
   - 또는 Chocolatey: `choco install sbt`
   - 확인: `sbt --version`

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

### 수동 빌드
```bash
cd UI/scala-search
sbt assembly
```

빌드가 성공하면 `target/scala-2.13/policy-search-engine.jar` 파일이 생성됩니다.

## 사용 방법

Python 애플리케이션에서 자동으로 사용됩니다. Scala 엔진이 사용 가능하면 우선 사용하고, 
사용 불가능하면 기존 Python FAISS 엔진으로 자동 폴백됩니다.

### 수동 테스트

```bash
# 입력 JSON 파일 생성
echo '{"type":"search","query":"임산부 의료비","region":"서울","topK":10}' > test_input.json

# Scala 엔진 실행
java -jar UI/scala-search/target/scala-2.13/policy-search-engine.jar \
  UI/synonyms.json \
  UI/policy_summary_langchain_streaming.csv \
  < test_input.json
```

## 문제 해결

### 1. JAR 파일을 찾을 수 없음
- `UI/scala-search` 폴더에서 `sbt assembly` 실행하여 빌드
- 빌드 후 `target/scala-2.13/policy-search-engine.jar` 파일 확인

### 2. Java를 찾을 수 없음
- Java JDK 설치 확인
- 환경 변수 PATH에 Java 경로 추가

### 3. sbt를 찾을 수 없음
- sbt 설치 확인
- 환경 변수 PATH에 sbt 경로 추가

### 4. CSV 파일을 찾을 수 없음
- `UI/policy_summary_langchain_streaming.csv` 파일 존재 확인
- 또는 `UI/scala-search/src/main/scala/com/policysearch/Main.scala`에서 경로 수정

## 성능

- Scala 엔진은 더 정확한 검색 결과를 제공하지만 초기 로딩 시간이 필요합니다
- 대량의 정책 데이터에서도 안정적으로 작동합니다
- 타입 안전성으로 런타임 오류를 방지합니다

## 폴백 동작

Scala 엔진이 사용 불가능한 경우:
- 자동으로 기존 Python FAISS 엔진으로 전환
- 로그에 경고 메시지 출력
- 검색 기능은 정상 작동

