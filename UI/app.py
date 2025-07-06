from __future__ import annotations

from flask import Flask, request, jsonify, render_template
import warnings, pandas as pd, re, json, pathlib
from datetime import date
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# ───────────────────────────────────────────────────────────
# 0) 기본 설정
# ───────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message="The parameter 'token_pattern'"
)
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "여성맞춤정책_분류+상세_v6.xlsx"
SYN_FILE = BASE_DIR / "synonyms.json"

app = Flask(__name__)

# ───────────────────────────────────────────────────────────
# 1) 동의어 사전(JSON) 로드
# ───────────────────────────────────────────────────────────
with open(SYN_FILE, encoding="utf-8") as f:
    synonym_expansion: dict[str, list[str]] = json.load(f)
_syn_lookup = {w: root for root, words in synonym_expansion.items() for w in words}

# ───────────────────────────────────────────────────────────
# 2) 형태소 전처리
# ───────────────────────────────────────────────────────────
okt = Okt()
STOP = {
    '또는','그리고','및','하지만','그러나','또한','따라서','즉','이하','이상','미만','초과','내외','정도','기준',
    '의한','의하여','의하면','통한','지원','감면','제외','포함','있음','있는','되어','대한','경우','시','후','중',
    '합니다','됩니다','안돼요','안됨','불가능','가능','있습니다','하시기','되어야','되어서','위해','대해','대상','대비','관련','관해'
}

def morph(text: str) -> str:
    return " ".join(tok for tok in okt.morphs(str(text)) if tok not in STOP)

# ───────────────────────────────────────────────────────────
# 3) 데이터 로드 & TF-IDF
# ───────────────────────────────────────────────────────────
COLS_FOR_TFIDF = ["지역","제목","신청방법","접수기관","지원대상","지원내용","문의처"]
_df_base = pd.read_excel(DATA_FILE, engine="openpyxl")
_base_text = _df_base[COLS_FOR_TFIDF].fillna("").agg(" ".join, axis=1)

_morph_vec = TfidfVectorizer(tokenizer=lambda s: s.split(), ngram_range=(1,2), lowercase=False).fit(_base_text)
_char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4), lowercase=False).fit(_base_text)

_docs = _base_text.tolist()
_bm25 = BM25Okapi([morph(d).split() for d in _docs])
_sbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
_doc_embs = _sbert.encode(_docs, convert_to_tensor=True)

def sim_mix(q: str, doc: str) -> float:
    vq = hstack((_morph_vec.transform([morph(q)]), _char_vec.transform([q])))
    vd = hstack((_morph_vec.transform([morph(doc)]), _char_vec.transform([doc])))
    return float(cosine_similarity(vq, vd)[0,0])

# ───────────────────────────────────────────────────────────
# 4) 나이·개월 계산, 범위 판정 및 age_ok
# ───────────────────────────────────────────────────────────
RE_Y_RANGE = re.compile(r'(?:만\s*)?(\d+)\s*세\s*[~\-]\s*(?:만\s*)?(\d+)\s*세')
RE_Y_SINGLE = re.compile(r'(?:만\s*)?(\d+)\s*세\s*(이상|초과|미만|이하)')
RE_M_RANGE = re.compile(r'(\d+)\s*개월\s*[~\-]\s*(\d+)\s*개월')
RE_M_SINGLE = re.compile(r'(\d+)\s*개월\s*(이상|초과|미만|이하)')

def calc_age(dob: str) -> tuple[int|None, int|None]:
    if not dob: return None, None
    try:
        y, m, d = map(int, re.split(r'\D+', dob)[:3])
        today = date.today()
        yrs = today.year - y - ((today.month, today.day) < (m, d))
        mos = (today.year - y)*12 + (today.month - m) - (1 if today.day < d else 0)
        return yrs, mos
    except: return None, None

def age_tags(yrs: int) -> set[str]:
    s = set()
    if yrs <= 17: s.add("아동")
    if 9 <= yrs <= 24: s.add("청소년")
    if 15 <= yrs <= 39: s.add("청년")
    if 15 <= yrs <= 49: s.add("가임기여성")
    if yrs >= 65: s.add("노인")
    return s

def age_ok(txt: str, yrs: int|None, mos: int|None) -> bool:
    if yrs is None: return True
    found = False
    for lo, hi in RE_Y_RANGE.findall(txt):
        found = True
        if int(lo) <= yrs <= int(hi): return True
    for v, c in RE_Y_SINGLE.findall(txt):
        found = True; v = int(v)
        if ((c=='이상' and yrs>=v) or (c=='초과' and yrs>v) or (c=='이하' and yrs<=v) or (c=='미만' and yrs<v)): return True
    if mos is not None:
        for lo, hi in RE_M_RANGE.findall(txt):
            found = True
            if int(lo) <= mos <= int(hi): return True
        for v, c in RE_M_SINGLE.findall(txt):
            found = True; v = int(v)
            if ((c=='이상' and mos>=v) or (c=='초과' and mos>v) or (c=='이하' and mos<=v) or (c=='미만' and mos<v)): return True
    return not found

# ───────────────────────────────────────────────────────────
# 5) 검색 유틸
# ───────────────────────────────────────────────────────────
def _as_list(v):
    if v is None: return []
    if isinstance(v, list): return [x.strip() for x in v if x.strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]

# ───────────────────────────────────────────────────────────
# 6) search_policies()
# ───────────────────────────────────────────────────────────
def search_policies(p: dict) -> pd.DataFrame:
    # 1) 원본 복사 + __cats 한 번만 생성 (공백 제거)
    df = _df_base.copy()
    df["__cats"] = (
        df["카테고리_분류"].fillna("").apply(
            lambda v: {x.strip().replace(" ", "") for x in v.split(",") if x.strip()}
        )
    )

    # 2) 파라미터 파싱 & 정규화
    regions   = [r.strip() for r in _as_list(p.get("region"))]
    cats      = [c.replace(" ", "")      for c in _as_list(p.get("category"))]
    supports  = _as_list(p.get("support"))
    kw        = p.get("kw_text", "").strip()
    age_y, age_m = calc_age(p.get("dob", ""))

    # 3) 아무 필터·키워드 없이 단순 조회만 할 때 → 전체 반환
    if ((not regions or (len(regions) == 1 and regions[0] == "전국"))
        and not cats and not supports
        and age_y is None and kw == ""):
        full = _df_base.copy()
        full["combined_score"] = 0.0
        return full.reset_index()  # 'index' 컬럼 확보

    # 4) 나이 태그 OR (예: 24세 → "청년" 등)
    if age_y is not None:
        df["__cats"] = df["__cats"].apply(lambda s: s.union(age_tags(age_y)))

    # 5) 지역 OR 필터 — regions가 있을 때만 적용, '전국'은 스킵
    if regions and not (len(regions) == 1 and regions[0] == "전국"):
        df = df[df["지역"].fillna("").apply(
            lambda v: any(v.startswith(r) for r in regions)
        )]

    # 6) 카테고리 OR 필터
    if cats:
        df = df[df["__cats"].apply(lambda s: bool(set(cats) & s))]

    # 7) 지원형태 OR 필터
    if supports:
        df = df[df["지원형태_분류"].fillna("").apply(
            lambda v: bool(set(x.strip() for x in v.split(",")) & set(supports))
        )]

    # 8) 본문(age_ok) 필터
    if age_y is not None:
        agg = df[COLS_FOR_TFIDF].fillna("").agg(" ".join, axis=1)
        df = df[agg.apply(lambda t: age_ok(t, age_y, age_m))]

    # 9) 빈 DataFrame 검사: 비어 있으면 전체 데이터로 바로 반환
    if df.empty:
        full = _df_base.copy()
        full["combined_score"] = 0.0
        return full.reset_index()  # 'index' 컬럼 확보

    # 10) 키워드 기반 스코어링 + 동의어 매칭
    # 10-1) TF/char 혼합 유사도
    tf_scores = df[COLS_FOR_TFIDF].fillna("").agg(" ".join, axis=1) \
                     .apply(lambda d: sim_mix(kw, d)).to_numpy()
    # 10-2) BM25 & SBERT
    full_bm25  = _bm25.get_scores(morph(kw).split())
    full_sbert = util.cos_sim(
                    _sbert.encode(kw, convert_to_tensor=True),
                    _doc_embs
                  )[0].cpu().numpy()
    idxs       = df.index.to_list()
    bm25_scores  = full_bm25[idxs]
    sbert_scores = full_sbert[idxs]
    # 10-3) 정규화 및 가중합
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)
    df["combined_score"] = (
        0.2 * norm(tf_scores) +
        0.3 * norm(bm25_scores) +
        0.5 * norm(sbert_scores)
    )
    # 10-4) 동의어 매칭 표시
    if (root := _syn_lookup.get(kw)):
        patt = re.compile("|".join(map(re.escape, synonym_expansion[root])))
        agg  = df[COLS_FOR_TFIDF].fillna("").agg(" ".join, axis=1)
        df["syn_match"] = agg.str.contains(patt)
    else:
        df["syn_match"] = False
    # 10-5) 0점 문서 제거 및 정렬
    df = df[df["syn_match"] | (df["combined_score"] > 0)]
    df = df.assign(__syn=df["syn_match"].astype(int))
    df = df.sort_values(
        by=["__syn", "combined_score"],
        ascending=[False, False]
    )
    df.drop(columns=["__syn", "syn_match"], inplace=True)

    # 11) 인덱스 재설정 → 카테고리 재매핑 → __cats 드롭 → 반환
    df = df.reset_index()  # 기존 행 인덱스가 'index' 컬럼으로 남습니다
    df["카테고리_분류"] = df["__cats"].apply(lambda s: ",".join(sorted(s)))
    df.drop(columns="__cats", inplace=True)

    return df



# ───────────────────────────────────────────────────────────
# 7) Flask 라우트
# ───────────────────────────────────────────────────────────
@app.route("/")
def home(): return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    params = request.get_json(force=True) or {}
    kw     = params.get("kw_text", "").strip()

    # 1) 필터링 결과 가져오기
    df = search_policies(params)

    # 2) 인덱스를 'index' 컬럼으로 남기기
    df = df.reset_index()   # drop=True 생략!

    # 3) 반환할 컬럼 목록에 'index' 포함
    cols = ["index","지역","제목","카테고리_분류", "지원형태_분류", ] + (["combined_score"] if kw else [])

    # 4) JSON 응답
    return jsonify(df[cols].to_dict("records"))

@app.route("/detail/<int:idx>")
def detail(idx: int):
    row = _df_base.iloc[idx]
    hide = {"지원형태_분류","카테고리_분류","기타",""}
    item = {k:v for k,v in row.items() if k not in hide and pd.notna(v)}
    table_html = None
    fn = row.get("기타")
    if pd.notna(fn):
        td = BASE_DIR/'fold'
        for p in [td/fn] + [td/f"{fn}{e}" for e in ('.xlsx','.xls','.csv')]:
            if p.exists():
                df_t = pd.read_excel(p, engine='openpyxl') if p.suffix in ('.xlsx','.xls') else pd.read_csv(p)
                table_html = df_t.to_html(classes="table table-striped", index=False, border=0)
                break
    return render_template("detail.html", item=item, table_html=table_html)

if __name__ == "__main__":
    app.run(debug=True)
