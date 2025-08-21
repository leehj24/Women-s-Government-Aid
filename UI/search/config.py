# config.py
MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = ".policy_cache"              # FAISS/임베딩 캐시 폴더
REGION_COL = "지역"                      # 지역 컬럼명
REGION_MODE = "filter"                   # "filter" | "boost"
THRESHOLD = 0.0                          # 유사도 임계값(필요시 상향)
TOPK_CANDIDATES = 200                    # 최대 반환 후보 수

# 텍스트 컬럼: 없으면 textprep가 생성 ("검색본문_nat")
TEXT_COLS = ("검색본문_nat",)

# 데이터 파일 환경변수 키
DATA_ENV = "POLICY_XLSX"

# 파일 탐색 후보(상대/상위 경로)
DEFAULT_FILE_CANDIDATES = [
    "여성맞춤정책_요약_2차_결과_병합.xlsx",
    "../여성맞춤정책_요약_2차_결과_병합.xlsx",
]
