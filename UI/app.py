from __future__ import annotations

from flask import Flask, request, jsonify, render_template
import pandas as pd, re, pathlib
from typing import List
from search.policy_search import find_policies, get_base_df

BASE_DIR = pathlib.Path(__file__).resolve().parent
app = Flask(__name__)

# detail 렌더링용 원본 DF (policy_search와 동일 원본)
_df_base = get_base_df()


def _as_list(v) -> List[str]:
    """문자열 'a,b,c' 또는 리스트 -> ['a','b','c']로 표준화"""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    p = request.get_json(force=True) or {}

    kw       = (p.get("kw_text") or "").strip()
    region   = (p.get("region") or "").strip()
    dob      = (p.get("dob") or "").strip() or None
    cats_in  = _as_list(p.get("category"))
    sups_in  = _as_list(p.get("support"))

    # 전부 policy_search로 위임 (점수 컬럼은 반환하지 않도록 policy_search에서 제거됨)
    df = find_policies(
        input=kw,
        region=region,
        dob=dob,
        categories=cats_in,
        supports=sups_in,
        out="dataframe",
    )

    # 라벨 컬럼 정규화(프론트가 기대하는 이름으로 맞춤)
    if "카테고리_분류" not in df.columns and "category_label" in df.columns:
        df["카테고리_분류"] = df["category_label"]
    if "지원형태_분류" not in df.columns and "support_label" in df.columns:
        df["지원형태_분류"] = df["support_label"]

    # 프론트에서 쓰는 최소 컬럼 구성 (score는 절대 포함하지 않음)
    cols = ["orig_index", "지역", "제목", "카테고리_분류", "지원형태_분류"]
    cols = [c for c in cols if c in df.columns]

    payload = (
        df.reset_index(drop=True)[cols]
          .rename(columns={"orig_index": "index"})  # detail/<index> 에서 사용
          .to_dict("records")
    )
    return jsonify(payload)


@app.route("/detail/<int:idx>")
def detail(idx: int):
    # idx는 policy_search에서 넘겨준 orig_index
    row = _df_base.iloc[idx]

    # 상세 본문에서 감출 컬럼
    hide = {
        "대상유형", "제목", "지원형태_분류", "카테고리_분류", "기타", "orig_index",
        "age_eff_ranges","age_has_rule",
        "지원대상", "지원내용",
        "지원대상_원문", "지원대상_초벌요약",
        "지원내용_원문", "지원내용_초벌요약",
    }

    # 일반 필드(표시용): hide 제외하고 NaN 제거
    fields = []
    for k, v in row.items():
        if k in hide:
            continue
        if pd.notna(v):
            fields.append((k, v))

    def _norm(x):
        if pd.isna(x) or x is None:
            return ""
        s = str(x)
        s = re.sub(r"\r\n?", "\n", s)
        return s.strip()

    # 제목은 카드 맨 위에 크게 보여주기 위해 별도 변수로 전달
    title = _norm(row.get("제목"))

    # 지원대상/내용(원문/요약) 표시에 필요한 값 정리
    tgt_orig = _norm(row.get("지원대상_원문"))
    tgt_sum  = _norm(row.get("지원대상_초벌요약"))
    cnt_orig = _norm(row.get("지원내용_원문"))
    cnt_sum  = _norm(row.get("지원내용_초벌요약"))

    tgt_same = (tgt_orig and tgt_sum and tgt_orig == tgt_sum) or (tgt_orig and not tgt_sum)
    cnt_same = (cnt_orig and cnt_sum and cnt_orig == cnt_sum) or (cnt_orig and not cnt_sum)

    tgt_display = tgt_orig if tgt_same else (tgt_sum or tgt_orig)
    cnt_display = cnt_orig if cnt_same else (cnt_sum or cnt_orig)

    has_toggle = ((tgt_orig and tgt_sum and tgt_orig != tgt_sum) or
                  (cnt_orig and cnt_sum and cnt_orig != cnt_sum))

    # '기타' 첨부 표 로딩
    table_html = None
    fn = row.get("기타")
    if pd.notna(fn):
        td = BASE_DIR / "fold"
        for p in [td / fn] + [td / f"{fn}{e}" for e in (".xlsx", ".xls", ".csv")]:
            if p.exists():
                df_t = (
                    pd.read_excel(p, engine="openpyxl")
                    if p.suffix.lower() in (".xlsx", ".xls")
                    else pd.read_csv(p)
                )
                def _fmt(x):
                    s = "" if pd.isna(x) else str(x)
                    s = s.replace("\\r\\n", "\n").replace("\\n", "\n")
                    return s.strip()
                df_t = df_t.applymap(_fmt)
                table_html = df_t.to_html(classes="ext-table", index=False, border=0, escape=True)
                break

    return render_template(
        "detail.html",
        title=title,                   # ← 제목을 템플릿에 별도로 전달 (상단에 크게 배치)
        fields=fields,
        # 표시 텍스트
        target_display=tgt_display,
        content_display=cnt_display,
        # 원문/요약 원본값
        target_orig=tgt_orig, target_sum=tgt_sum,
        content_orig=cnt_orig, content_sum=cnt_sum,
        # 동일 여부 & 토글
        target_same=tgt_same, content_same=cnt_same,
        has_toggle=has_toggle,
        table_html=table_html,
    )


if __name__ == "__main__":
    app.run(debug=True)
