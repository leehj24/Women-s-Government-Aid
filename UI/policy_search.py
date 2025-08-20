# -*- coding: utf-8 -*-
# policy_search.py — 여성 정책 검색/추천 엔진 (score 제거 + 쿼리 기반 나이필터 통합)

import os, re, time, json, argparse, hashlib
from datetime import datetime, date
from typing import List, Optional, Literal, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer
import faiss, torch

# ───────── 경로/파일 ─────────
HERE = Path(__file__).resolve().parent

CACHE_DIR = Path(os.getenv("POLICY_CACHE_DIR", ".policy_cache")).expanduser()
if not CACHE_DIR.is_absolute():
    CACHE_DIR = HERE / CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_ENV = os.getenv("POLICY_XLSX", "")
CANDIDATES = [
    Path(DATA_ENV).expanduser(),
    HERE / "여성맞춤정책_요약_2차_결과_병합.xlsx",
    HERE.parent / "여성맞춤정책_요약_2차_결과_병합.xlsx",
]

FILE_PATH: Optional[Path] = None
for p in CANDIDATES:
    if p and str(p).strip() and p.is_file():
        FILE_PATH = p
        break
if FILE_PATH is None:
    hits = list(HERE.rglob("여성맞춤정책_요약_2차_결과_병합.xlsx"))
    if hits:
        FILE_PATH = hits[0]
if FILE_PATH is None:
    tried = "\n  - " + "\n  - ".join(str(p) for p in CANDIDATES)
    raise FileNotFoundError(
        "[데이터 파일을 찾을 수 없음]\n다음 위치를 확인하세요:\n"
        f"{tried}\n또는 환경변수 POLICY_XLSX에 정확한 경로를 설정하세요."
    )
FILE_PATH = FILE_PATH.resolve()
print(f"✅ 데이터 파일: {FILE_PATH}")
print(f"✅ 캐시 디렉터리: {CACHE_DIR}")

# ───────── 데이터 로드 ─────────
df_raw = pd.read_excel(FILE_PATH)
# detail 라우팅용 원본 인덱스 보존
df_raw = df_raw.reset_index(drop=False).rename(columns={'index': 'orig_index'})
# 제목 없는 행 제거
df = df_raw.dropna(subset=["제목"]).copy()

# ───────── 전처리(검색본문) ─────────
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = re.sub(r"(?m)^\s*([\-–—\u2010-\u2015\u2212\u2043\u2022\u25CB\u25CF\u25AA\u25A0\u30FB]|[0-9]+[.)])\s+", " ", t)
    t = re.sub(r"[\u2460-\u2473\u3251-\u325F\u32B1-\u32BF]", " ", t)
    t = re.sub(r"[※＊*]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def clean_field_value(val: str, field_name: str) -> str:
    if not isinstance(val, str): return ""
    v = re.sub(rf"^\s*{re.escape(field_name)}\s*[:\-–—]?\s*", "", val.strip())
    return normalize_text(v)

def unify_text_natural(row: pd.Series) -> str:
    title   = normalize_text(row.get("제목",""))
    region  = normalize_text(row.get("지역",""))
    target  = clean_field_value(row.get("지원대상",""), "지원대상")
    content = clean_field_value(row.get("지원내용",""), "지원내용")
    parts = []
    if title:   parts.append(f"정책명은 {title}")
    if region:  parts.append(f"지역은 {region}")
    if target:  parts.append(f"지원대상은 {target}")
    if content: parts.append(f"지원내용은 {content}")
    if not parts: return ""
    if len(parts) == 1:  return parts[0] + "이다."
    if len(parts) == 2:  return parts[0] + "이고, " + parts[1] + "이다."
    return "이고, ".join(parts[:-1]) + "이며, " + parts[-1] + "이다."

TEXT_COL = "검색본문_nat"
df[TEXT_COL] = df.apply(unify_text_natural, axis=1)

# ───────── 컬럼 자동탐지 ─────────
def _norm_colname(s: str) -> str:
    return re.sub(r"[\s_/()\-·.,]+", "", s or "").lower()

def _find_col(df_in: pd.DataFrame, prefer: List[str], fuzzy: List[str]) -> Optional[str]:
    for c in prefer:
        if c in df_in.columns:
            return c
    nmap = {c: _norm_colname(c) for c in df_in.columns}
    want = [_norm_colname(x) for x in prefer+fuzzy]
    for c, nc in nmap.items():
        if any(w in nc for w in want):
            return c
    return None

CATEGORY_COL = _find_col(df, ['카테고리_분류','category_label'], ['카테고리','category','label'])
SUPPORT_COL  = _find_col(df, ['지원형태_분류','support_label'], ['지원형태','support','label'])

# ───────── 임베딩/FAISS ─────────
MODEL_NAME = "nlpai-lab/KURE-v1"
device = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"✅ 모델 로드: {MODEL_NAME} (device={device})")
model = SentenceTransformer(MODEL_NAME, device=device)

def _corpus_digest(corpus: List[str]) -> str:
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode("utf-8"))
    h.update(str(len(corpus)).encode("utf-8"))
    for line in corpus:
        if not isinstance(line, str):
            line = "" if pd.isna(line) else str(line)
        h.update(line.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()

def _safe_part(s: str) -> str:
    s = str(s)
    s_ascii = s.encode("ascii", "ignore").decode("ascii")
    s_ascii = re.sub(r"[^A-Za-z0-9._-]+", "_", s_ascii).strip("_.")
    return s_ascii or "corpus"

def _cache_paths(corpus: List[str]) -> Tuple[str, str, str]:
    base   = os.path.splitext(os.path.basename(str(FILE_PATH)))[0]
    digest = _corpus_digest(corpus)[:16]
    base_s  = _safe_part(base)
    model_s = _safe_part(MODEL_NAME.replace("/", "_"))
    key = f"{base_s}.{len(corpus)}.{model_s}.{digest}"
    if len(key) > 120:
        key = f"{base_s[:30]}.{len(corpus)}.{model_s[:30]}.{digest}"
    faiss_path = str(CACHE_DIR / f"{key}.faiss")
    npy_path   = str(CACHE_DIR / f"{key}.npy")
    meta_path  = str(CACHE_DIR / f"{key}.json")
    return faiss_path, npy_path, meta_path

def _save_cache(index: faiss.Index, embeddings: np.ndarray, meta_path: str, faiss_path: str, npy_path: str):
    faiss.write_index(index, faiss_path)
    np.save(npy_path, embeddings)
    meta = {
        "model": MODEL_NAME, "ntotal": int(index.ntotal),
        "embedding_dim": int(embeddings.shape[1]),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "file": os.path.basename(str(FILE_PATH)), "text_col": TEXT_COL,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _load_cache(faiss_path: str, npy_path: str) -> Tuple[faiss.Index, np.ndarray]:
    index = faiss.read_index(faiss_path)
    embeddings = np.load(npy_path)
    return index, embeddings

def _build_or_load_index(corpus: List[str], force_rebuild: bool = False) -> Tuple[faiss.Index, np.ndarray]:
    faiss_path, npy_path, meta_path = _cache_paths(corpus)
    if (not force_rebuild) and os.path.exists(faiss_path) and os.path.exists(npy_path):
        try:
            index, embeddings = _load_cache(faiss_path, npy_path)
            if index.ntotal != len(corpus):
                raise RuntimeError("캐시 ntotal 불일치")
            print(f"⚡ 캐시 로드 완료: {faiss_path}")
            return index, embeddings
        except Exception as e:
            print(f"캐시 로드 실패, 재생성합니다. 이유: {e}")

    print("✅ 정책 임베딩 생성...")
    t0 = time.time()
    embeddings = model.encode(corpus, convert_to_numpy=True, batch_size=64, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"✅ 임베딩 완료: {time.time()-t0:.2f}s, 벡터수={len(embeddings)}")

    try:
        _save_cache(index, embeddings, meta_path, faiss_path, npy_path)
        print(f"💾 캐시 저장: {faiss_path}")
    except Exception as e:
        print(f"캐시 저장 실패(무시): {e}")

    return index, embeddings

FORCE_REBUILD = os.getenv("POLICY_REBUILD", "0") in ("1","true","True","YES","yes")
corpus = df[TEXT_COL].fillna("").tolist()
index, _embeddings = _build_or_load_index(corpus, force_rebuild=FORCE_REBUILD)

# ───────── 나이 필터 (신규: 쿼리/age_eff_ranges 지원) ─────────
# 쿼리에서 나이 구간 뽑기 (예: "만 20~29세", "20대", "30세 이상/미만")
AGE_RANGE_RE  = re.compile(r"(?:만\s*)?(\d{1,3})\s*[~\-]\s*(?:만\s*)?(\d{1,3})\s*세")
AGE_SINGLE_RE = re.compile(r"(?:만\s*)?(\d{1,3})\s*세\s*(이상|초과|이하|미만)?")

def parse_query_age(text: str) -> List[Tuple[int,int]]:
    out: List[Tuple[int,int]] = []
    if not text:
        return out
    for a,b in AGE_RANGE_RE.findall(text):
        a,b = int(a), int(b)
        lo,hi = min(a,b), max(a,b)
        out.append((lo,hi))
    for n,b in AGE_SINGLE_RE.findall(text):
        n = int(n)
        if not b: out.append((n,n))
        elif b in ("이상","초과"): out.append((n + (b=="초과"), 200))
        elif b in ("이하","미만"): out.append((0, n - (b=="미만")))
    m = re.search(r"(\d{1,2})\s*대", text)
    if m:
        d = int(m.group(1)); out.append((d*10, d*10+9))
    return out

def _ranges_intersect(a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])

def age_match_from_json(policy_ranges_str: Optional[str], req_ranges: List[Tuple[int,int]]) -> bool:
    """엑셀에 age_eff_ranges(JSON 문자열)가 있을 때 사용. 없으면 True."""
    try:
        pr = json.loads(policy_ranges_str) if policy_ranges_str else []
    except Exception:
        pr = []
    if not pr:
        pr = [(0,200)]
    if not req_ranges:
        return True
    for lo1,hi1 in pr:
        for lo2,hi2 in req_ranges:
            if _ranges_intersect((lo1,hi1),(lo2,hi2)):
                return True
    return False

# (기존) 텍스트에서 나이 규칙 추출
KW_RANGES = [
    (re.compile(r"가임기\s*여성|가임기여성"), (15,49)),
    (re.compile(r"청소년"), (13,20)),
    (re.compile(r"아동"), (0,12.999)),
    (re.compile(r"어린이"), (3,13)),
    (re.compile(r"노인|고령자|어르신"), (65,float("inf"))),
]

def _norm_min(s):
    if not isinstance(s,str): return ""
    return re.sub(r"\s{2,}"," ", s.replace("\r"," ").replace("\n"," ").replace("\t"," ")).strip()

def extract_age_constraints(t):
    t = _norm_min(t or ""); cons=[]
    for m in re.finditer(r"(?:만\s*)?(\d+)\s*세\s*이상\s*[~\-–]\s*(?:만\s*)?(\d+)\s*세\s*미만", t):
        lo,hi=int(m[1]),int(m[2]); cons.append((lo,hi,True,False))
    for m in re.finditer(r"만\s*(\d+)\s*세\s*이상\s*만\s*(\d+)\s*세\s*이하", t):
        lo,hi=int(m[1]),int(m[2]); cons.append((lo,hi,True,True))
    for m in re.finditer(r"(?:만\s*)?(\d+)\s*세\s*[~\-–]\s*(?:만\s*)?(\d+)\s*세", t):
        lo,hi=int(m[1]),int(m[2]); cons.append((lo,hi,True,True))
    for m in re.finditer(r"(?:만\s*)?(\d+)\s*[~\-–]\s*(?:만\s*)?(\d+)\s*세", t):
        lo,hi=int(m[1]),int(m[2]); cons.append((lo,hi,True,True))
    for m in re.finditer(r"(?:만\s*)?(\d+)\s*세\s*(이상|이하|미만|초과|세이상|세이하|세미만)", t):
        v=int(m[1]); typ=m[2]
        if typ in ("이상","세이상"): cons.append((v,None,True,None))
        elif typ in ("이하","세이하"): cons.append((None,v,None,True))
        elif typ in ("미만","세미만"): cons.append((None,v,None,False))
        elif typ=="초과": cons.append((v,None,False,None))
    for m in re.finditer(r"생후\s*(\d+)\s*개월\s*[~\-–]\s*(?:만\s*)?(\d+)\s*세", t):
        lo=float(m[1])/12.0; hi=int(m[2]); cons.append((lo,hi,True,True))
    return cons

def extract_kw_constraints(t):
    t=_norm_min(t or ""); out=[]
    for pat,(lo,hi) in KW_RANGES:
        if pat.search(t): out.append((float(lo), float(hi), True, True))
    return out

def _policy_text_matches_req_ranges(text: str, req_ranges: List[Tuple[int,int]]) -> bool:
    """엑셀에 age_eff_ranges 없을 때, '지원대상' 텍스트에서 구간 추출해 비교."""
    if not req_ranges:
        return True
    cons = extract_age_constraints(text)
    kwc  = extract_kw_constraints(text)
    # cons/kwc → 대표 구간(lo, hi) 집약
    ranges = []
    if cons:
        los=[c[0] for c in cons if c[0] is not None]
        his=[c[1] for c in cons if c[1] is not None]
        lo = max(los) if los else 0
        hi = min(his) if his else 200
        ranges.append((float(lo), float(hi)))
    if kwc:
        for lo,hi,_,_ in kwc:
            ranges.append((float(lo), float(hi)))
    if not ranges:   # 신호 없으면 제한 없음
        return True
    for plo,phi in ranges:
        for rlo,rhi in req_ranges:
            if _ranges_intersect((plo,phi),(rlo,rhi)): return True
    return False

# 생년월일 → 나이(년)
def parse_birthdate(s: str):
    if not s: return None
    for fmt in ("%Y-%m-%d","%Y.%m.%d","%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except:
            pass
    try:
        return parser.parse(s.strip()).date()
    except:
        return None

def calc_age_years_precise(birth, ref=None):
    if not birth: return None
    if ref is None: ref = date.today()
    return (ref - birth).days/365.2425

# ───────── 라벨(카테고리/지원형태) 부분일치 ─────────
_norm_label_re = re.compile(r"[\s/()\[\]{}·.,\-]+")
def _norm_label(s: str) -> str:
    return _norm_label_re.sub("", s or "").lower()

def _expand_queries(queries: List[str]) -> List[str]:
    out = set()
    for q in queries or []:
        q = (str(q) if q is not None else "").strip()
        if not q: continue
        out.add(q)
        for tok in re.split(r"[\/,|]", q):
            tok = tok.strip()
            if tok: out.add(tok)
        no_paren_chars    = re.sub(r"[()]", "", q)          # 대학(원)생→대학원생
        no_paren_content  = re.sub(r"\([^)]*\)", "", q)      # 대학(원)생→대학생
        out.update([no_paren_chars, no_paren_content])
        def _nz(s): return _norm_label(s) if s else s
        out.update({_nz(q), _nz(no_paren_chars), _nz(no_paren_content)})
        SYN = {
            "대학(원)생": ["대학생", "대학원생"],
            "한부모": ["한부모가족", "한부모가정"],
            "고령자": ["노인", "어르신"],
            "1인가구": ["1인 가구", "독거"],
            "임신/출산/육아": ["임신", "출산", "육아", "임신출산육아"],
        }
        if q in SYN:
            out.update(SYN[q]); out.update(_nz(x) for x in SYN[q])
    return [x for x in out if x]

def _contains_any_substring(cell: str, queries: List[str]) -> bool:
    s = str(cell) if cell is not None else ""
    ex = _expand_queries(queries)
    if any(q for q in ex):
        if any(q in s for q in ex):    # 원문 부분일치
            return True
    s_norm = _norm_label(s)
    return any((qn and (qn in s_norm)) for qn in ex)

# ───────── 1차 필터 ─────────
def build_stage1_mask(df_in: pd.DataFrame,
                      categories: Optional[List[str]]=None,
                      supports: Optional[List[str]]=None,
                      region: Optional[str]=None,
                      birthdate: Optional[str]=None,
                      kw_text: Optional[str]=None) -> np.ndarray:
    n = len(df_in)
    mask = np.ones(n, dtype=bool)

    # 카테고리/지원형태
    if categories:
        col = _find_col(df_in, ['카테고리_분류','category_label'], ['카테고리','category','label'])
        if col:
            mask &= df_in[col].apply(lambda x: _contains_any_substring(x, categories)).to_numpy()
    if supports:
        col = _find_col(df_in, ['지원형태_분류','support_label'], ['지원형태','support','label'])
        if col:
            mask &= df_in[col].apply(lambda x: _contains_any_substring(x, supports)).to_numpy()

    # 지역: "" 또는 "전국"이면 미적용, 그 외 startswith
    r = (region or "").strip()
    if r and r != "전국" and "지역" in df_in.columns:
        mask &= df_in["지역"].astype(str).str.startswith(r, na=False).to_numpy()

    # 나이 필터: dob → 개인나이, 없으면 kw_text에서 구간 파싱
    req_ranges: List[Tuple[int,int]] = []
    if birthdate:
        b = parse_birthdate(birthdate)
        if b:
            age = int(calc_age_years_precise(b) or 0)
            req_ranges = [(age, age)]
    if (not req_ranges) and kw_text:
        req_ranges = parse_query_age(kw_text)

    if req_ranges:
        if "age_eff_ranges" in df_in.columns:
            mask &= df_in["age_eff_ranges"].apply(lambda s: age_match_from_json(s, req_ranges)).to_numpy()
        elif "지원대상" in df_in.columns:
            mask &= df_in["지원대상"].apply(lambda t: _policy_text_matches_req_ranges(t, req_ranges)).to_numpy()
        # else: 해당 컬럼 없으면 필터 스킵

    return mask

def filter_by_user_inputs(df_in: pd.DataFrame,
                          region: Optional[str],
                          dob: Optional[str],
                          categories: Optional[List[str]],
                          supports: Optional[List[str]],
                          kw_text: Optional[str]=None) -> pd.DataFrame:
    mask = build_stage1_mask(
        df_in, categories=categories, supports=supports,
        region=region, birthdate=dob, kw_text=kw_text
    )
    return df_in[mask].copy()

# ───────── 시맨틱 검색 ─────────
def _faiss_search(query: str, top_k: Optional[int] = None):
    total = int(index.ntotal)
    k = total if (top_k is None or top_k <= 0 or top_k > total) else int(top_k)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k=k)
    return D[0], I[0]

def semantic_search(query: str, top_k: Optional[int] = None,
                    out: Literal["dataframe","json","csv"]="dataframe"):
    if not query or not str(query).strip():
        return pd.DataFrame() if out == "dataframe" else "[]"
    _, I = _faiss_search(str(query).strip(), top_k)
    rows = df.iloc[I].reset_index(drop=True).copy()
    return _format_output(rows, out)

# ───────── 추천(쿼리 없음) ─────────
def _last_date_from_period(s: str):
    if not isinstance(s,str) or not s.strip():
        return pd.NaT
    s2 = s.replace("~","-").replace("–","-").replace("—","-")
    parts = re.findall(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", s2)
    try:
        return parser.parse(parts[-1]) if parts else pd.NaT
    except:
        return pd.NaT

def recommend(region: Optional[str]="전국",
              dob: Optional[str]=None,
              categories: Optional[List[str]]=None,
              supports: Optional[List[str]]=None,
              out: Literal["dataframe","json","csv"]="dataframe"):
    filtered = filter_by_user_inputs(df, region, dob, categories or [], supports or [], kw_text=None)
    tmp = filtered.copy()
    if "신청기간" in tmp.columns:
        tmp["_end"] = tmp["신청기간"].apply(_last_date_from_period)
        tmp = tmp.sort_values(by=["_end","제목"], ascending=[True,True]).drop(columns=["_end"])
    return _format_output(tmp.reset_index(drop=True), out)

# ───────── 통합 진입점 ─────────
def find_policies(input: str = "",
                  topk: Optional[int] = None,
                  region: Optional[str] = "전국",
                  dob: Optional[str] = None,
                  categories: Optional[List[str]] = None,
                  supports: Optional[List[str]] = None,
                  out: Literal["dataframe","json","csv"]="dataframe"):
    """
    - input 비었으면: 컬럼 기반 필터만 적용(전체 가능)
    - input 있으면: 시맨틱 검색 순서 유지 + 컬럼 필터 교집합 (점수 컬럼 없음)
    - 나이 필터: dob가 있으면 개인 나이 기준, 없으면 input에서 나이 구간 파싱
    """
    if not input or not str(input).strip():
        return recommend(region=region, dob=dob, categories=categories, supports=supports, out=out)

    _, I = _faiss_search(str(input).strip(), topk)
    # 쿼리에서 나이 구간 추출해서 필터에 전달
    mask = build_stage1_mask(df, categories or [], supports or [], region, dob, kw_text=input)
    kept = [i for i in I if mask[i]]
    if not kept:
        kept = list(I)

    pos = {i: p for p, i in enumerate(I)}
    kept_sorted = sorted(kept, key=lambda i: pos[i])
    rows = df.iloc[kept_sorted].reset_index(drop=True).copy()
    return _format_output(rows, out)

# ───────── 출력 포맷 ─────────
def _format_output(df_out: pd.DataFrame, out: Literal["dataframe","json","csv"]):
    if out == "dataframe":
        return df_out
    if out == "json":
        cols = [c for c in [
            "orig_index","제목","지역",
            "카테고리_분류","category_label",
            "지원형태_분류","support_label",
            "지원형태","신청기간","신청방법","접수기관",
            "지원대상","지원내용","문의처","기타","detail_url"
        ] if c in df_out.columns]
        return json.dumps(df_out[cols].to_dict(orient="records"), ensure_ascii=False, indent=2)
    if out == "csv":
        return df_out.to_csv(index=False)
    raise ValueError("out must be one of {'dataframe','json','csv'}")

# ───────── Flask에서 원본 DF ─────────
def get_base_df() -> pd.DataFrame:
    return df_raw.copy()

# (옵션) CLI
def _parse_list(s: Optional[str]) -> List[str]:
    if not s: return []
    return [x.strip() for x in re.split(r"[;,/|,]", s) if x.strip()]

def main(argv: Optional[List[str]] = None):
    import sys
    ap = argparse.ArgumentParser(description="여성 정책 검색/추천 엔진")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_find = sub.add_parser("find")
    ap_find.add_argument("--input", type=str, default="")
    ap_find.add_argument("--topk", type=int, default=0)
    ap_find.add_argument("--region", type=str, default="전국")
    ap_find.add_argument("--dob", type=str, default=None)
    ap_find.add_argument("--categories", type=str, default=None)
    ap_find.add_argument("--supports", type=str, default=None)
    ap_find.add_argument("--out", type=str, choices=["dataframe","json","csv"], default="json")
    ap_find.add_argument("--rebuild", action="store_true")

    if argv is None:
        argv = sys.argv[1:]
    args = ap.parse_args(argv)

    if getattr(args, "rebuild", False):
        global index, _embeddings
        print("🔄 캐시 재생성")
        index, _embeddings = _build_or_load_index(corpus, force_rebuild=True)

    res = find_policies(
        input=args.input, topk=(None if args.topk==0 else args.topk),
        region=args.region, dob=args.dob,
        categories=_parse_list(args.categories), supports=_parse_list(args.supports),
        out=args.out
    )
    if isinstance(res, pd.DataFrame):
        print(res.to_csv(index=False, sep="\t"))
    else:
        print(res)

if __name__ == "__main__":
    import sys
    in_ipy = ("ipykernel" in sys.modules) or ("IPython" in sys.modules)
    if not in_ipy:
        main()
