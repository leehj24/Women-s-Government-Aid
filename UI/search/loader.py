# loader.py
import os
from pathlib import Path
import pandas as pd
from typing import Tuple
from .config import DATA_ENV, DEFAULT_FILE_CANDIDATES
from .textprep import ensure_text_columns

SEARCH_PATTERNS = (
    "policy_summary_langchain_streaming.csv",
    "여성맞춤정책_요약_2차_결과_병합.xlsx",
)

def _resolve_file() -> Path:
    env = os.getenv(DATA_ENV, "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
    here = Path(__file__).resolve().parent
    for cand in DEFAULT_FILE_CANDIDATES:
        p = (here / cand).resolve()
        if p.is_file():
            return p
    for pat in SEARCH_PATTERNS:
        hits = list(here.rglob(pat))
        if hits:
            return hits[0].resolve()
    raise FileNotFoundError("정책 데이터파일을 찾을 수 없습니다. 환경변수 POLICY_XLSX를 설정하거나 파일을 같은 폴더에 두세요.")

def load_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    path = _resolve_file()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df_raw = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df_raw = pd.read_excel(path)
    else:
        raise ValueError(f"지원하지 않는 확장자입니다: {suffix}")
    df_raw = df_raw.reset_index(drop=False).rename(columns={'index': 'orig_index'})
    df = df_raw.dropna(subset=["제목"]).copy()
    df = ensure_text_columns(df, text_col_name="검색본문_nat")
    return df_raw, df, path
