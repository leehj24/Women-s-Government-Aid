// app.js

// ===== DOM refs =====
const personalCard        = document.getElementById("personalCard");
const globalCard          = document.getElementById("globalCard");
const personalForm        = document.getElementById("personalForm");
const globalSection       = document.getElementById("globalSection");
const filterForm          = document.getElementById("filterForm");
const supportSelect       = document.getElementById("supportSelect");
const btnFilterToggle     = document.getElementById("btnFilterToggle");
const btnFilterSearch     = document.getElementById("btnFilterSearch");
const customSummary       = document.getElementById("customSummary");
const btnEditInfo         = document.getElementById("btnEditInfo");
const customSearch        = document.getElementById("customSearch");
const btnSearch           = document.getElementById("btnSearch");
const btnSearchText       = document.getElementById("btnSearchText");
const btnSearchTextCustom = document.getElementById("btnSearchTextCustom");
const results             = document.getElementById("results");
const loading             = document.getElementById("loading");
const tags                = document.querySelectorAll(".tag");
const startCards          = document.querySelector(".start-cards");

// ===== Client-side paging (for '더 보기') =====
const PAGE_SIZE = 24;        // 한 번에 보여줄 카드 수
let lastResults = [];        // 서버에서 받은 전체 결과
let viewResults = [];        // 클라이언트 필터(지원형태 등) 적용 후 결과
let currentParams = {};      // 현재 조회 파라미터(지역/생년/카테고리/검색어 등)
let shownCount = 0;          // 지금까지 렌더링한 개수

// '더 보기' 버튼을 동적으로 만들어 결과 섹션 바로 아래에 붙인다
const loadMoreBtn = document.createElement("button");
loadMoreBtn.className = "btn";
loadMoreBtn.textContent = "더 보기";
loadMoreBtn.style.width = "100%";
loadMoreBtn.style.marginTop = "12px";
loadMoreBtn.style.display = "none"; // 기본 감춤
results.insertAdjacentElement("afterend", loadMoreBtn);

loadMoreBtn.addEventListener("click", () => {
  shownCount += PAGE_SIZE;
  renderResults(viewResults, { append: false }); // 전체 다시 렌더 (간단/안전)
});

// ===== 카테고리 표시 제한 유틸 =====
const CATEGORY_LIMIT  = 5;
const CATEGORY_SUFFIX = " 등";

function formatCategories(catStr, limit = CATEGORY_LIMIT, suffix = CATEGORY_SUFFIX) {
  const arr = (catStr || "").split(",").map(s => s.trim()).filter(Boolean);
  if (arr.length === 0) return "";
  if (arr.length <= limit) return arr.join(", ");
  return arr.slice(0, limit).join(", ") + suffix;
}

// title 속성 이스케이프
function escAttr(s) {
  return (s || "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// 나이 계산
function calculateAge(dob) {
  if (!dob) return null;
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (
    today.getMonth() < birth.getMonth() ||
    (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())
  ) {
    age--;
  }
  return age;
}

// ===== 초기 상태: 폼/결과 숨김 =====
[personalForm, globalSection, filterForm, customSummary, customSearch, results]
  .forEach(el => el.classList.add("hidden"));
loadMoreBtn.style.display = "none";

function showHomeView() {
  if (startCards) startCards.classList.remove("hidden");
  personalForm.classList.add("hidden");
  globalSection.classList.add("hidden");
  filterForm.classList.add("hidden");
  customSummary.classList.add("hidden");
  customSearch.classList.add("hidden");
  results.classList.add("hidden");
  results.innerHTML = "";
  loadMoreBtn.style.display = "none";
  currentParams = {};
  updateURL({}); // URL 초기화
}

const openPersonal = () => {
  if (startCards) startCards.classList.add("hidden");
  personalForm.classList.remove("hidden");
  globalSection.classList.add("hidden");
  filterForm.classList.add("hidden");
  customSummary.classList.add("hidden");
  customSearch.classList.add("hidden");
  results.classList.add("hidden");
  loadMoreBtn.style.display = "none";
  currentParams = {};
  updateURL({}); // URL 초기화
};

const openGlobal = () => {
  if (startCards) startCards.classList.add("hidden");
  globalSection.classList.remove("hidden");
  filterForm.classList.add("hidden");
  customSummary.classList.add("hidden");
  customSearch.classList.add("hidden");
  currentParams = {};
  // /all 경로에서 호출된 경우 경로 유지, 아니면 URL 업데이트 안 함
  // (fetchResults에서 자동으로 URL 업데이트됨)
  fetchResults({});
};

// ===== 시작 카드 이벤트 =====
if (personalCard) personalCard.onclick = openPersonal;
if (globalCard) globalCard.onclick = openGlobal;

// ===== 초기 라우트 모드 적용 =====
(() => {
  // URL 파라미터가 있으면 복원 시도
  if (restoreFromURL()) {
    return; // 복원 성공하면 여기서 종료
  }
  
  // URL 파라미터가 없으면 start_mode에 따라 초기화
  const mode = document.body.dataset.startMode || "home";
  if (mode === "personal") {
    openPersonal();
  } else if (mode === "global") {
    openGlobal();
  } else {
    showHomeView();
  }
})();

// ===== 조건검색 토글 =====
btnFilterToggle.onclick = () => filterForm.classList.toggle("hidden");

// ===== 맞춤정보 완료: 지역/생년/카테고리 포함 =====
btnSearch.onclick = () => {
  const region = document.getElementById("regionSelect").value;
  const dob    = document.getElementById("dob").value;
  const cats   = Array.from(tags)
    .filter(t => t.classList.contains("active"))
    .map(t => t.dataset.cat);
  const age = calculateAge(dob);

  currentParams = { region, dob, category: cats.join(",") };
  document.getElementById("summaryName").textContent = "내 맞춤 정보";
  let summaryLine = `${region} | ${cats.join(", ")}`;
  if (age !== null) summaryLine += ` | ${age}세`;
  document.getElementById("summaryLine").textContent = summaryLine;

  personalForm.classList.add("hidden");
  customSummary.classList.remove("hidden");
  customSearch.classList.remove("hidden");
  fetchResults(currentParams);
};

// ===== 정보수정 =====
btnEditInfo.onclick = () => {
  customSummary.classList.add("hidden");
  customSearch.classList.add("hidden");
  results.innerHTML = "";
  loadMoreBtn.style.display = "none";
  personalForm.classList.remove("hidden");
};

// ===== 검색 버튼들 =====
btnSearchText.onclick = () => {
  const kw = document.getElementById("kwText").value.trim();
  fetchResults({ kw_text: kw });
};

btnSearchTextCustom.onclick = () => {
  const kw = document.getElementById("kwTextCustom").value.trim();
  fetchResults({ ...currentParams, kw_text: kw });
};

// Enter키로도 검색 수행
["kwText", "kwTextCustom"].forEach(id => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (id === "kwText") btnSearchText.click();
        else btnSearchTextCustom.click();
      }
    });
  }
});

// ===== 태그 토글 =====
tags.forEach(t => t.addEventListener("click", () => t.classList.toggle("active")));

// ===== 지원형태 필터(클라이언트 측) =====
btnFilterSearch.onclick = () => {
  applySupportFilterAndRender();
};

function applySupportFilterAndRender() {
  const supArr = Array.from(supportSelect.selectedOptions).map(o => o.value);
  if (!supArr.length) {
    // 필터 없음 → 전체(서버 결과) 기준
    viewResults = [...lastResults];
  } else {
    viewResults = lastResults.filter(item => {
      const sups = (item["지원형태_분류"] || "")
        .split(",")
        .map(x => x.trim())
        .filter(Boolean);
      return supArr.some(s => sups.includes(s));
    });
  }
  shownCount = Math.min(PAGE_SIZE, viewResults.length);
  renderResults(viewResults, { append: false });
}

// ===== 결과 렌더링 =====
function renderResults(data, { append = false } = {}) {
  if (!append) {
    results.innerHTML = "";
  }

  if (!data.length) {
    results.innerHTML = "<p>검색 결과가 없습니다.</p>";
    loadMoreBtn.style.display = "none";
    results.classList.remove("hidden");
    return;
  }

  const toShow = data.slice(0, shownCount);
  // 현재 검색 파라미터를 URL에 포함하여 detail 페이지로 이동
  // 현재 페이지의 전체 URL을 backUrl로 사용 (검색 상태 유지)
  const currentUrl = new URL(window.location);
  const backUrl = currentUrl.pathname + (currentUrl.search || '');
  
  const html = toShow.map(item => {
    const fullCats = item["카테고리_분류"] || "";
    const catsText = formatCategories(fullCats);
    return `
      <div class="card result-card" onclick="location.href='/detail/${item.index}?back=${encodeURIComponent(backUrl)}'">
        <h3>${item.제목}</h3>
        <p class="cats" title="${escAttr(fullCats)}">${catsText}</p>
        <p>${item.지역}</p>
      </div>
    `;
  }).join("");

  results.innerHTML = html;

  // '더 보기' 버튼 표시/감춤
  if (shownCount < data.length) {
    loadMoreBtn.style.display = "";
    loadMoreBtn.textContent = `더 보기 (${shownCount} / ${data.length})`;
  } else {
    loadMoreBtn.style.display = "none";
  }

  results.classList.remove("hidden");
}

// ===== URL 파라미터 관리 =====
function updateURL(params) {
  const url = new URL(window.location);
  // 검색 파라미터가 있으면 URL에 추가, 없으면 제거
  if (params && Object.keys(params).length > 0) {
    Object.keys(params).forEach(key => {
      const value = params[key];
      if (value && value.toString().trim()) {
        url.searchParams.set(key, value);
      } else {
        url.searchParams.delete(key);
      }
    });
  } else {
    // 파라미터가 없으면 쿼리 스트링 제거
    url.search = '';
  }
  // 히스토리에 추가하지 않고 현재 URL만 업데이트 (뒤로가기 가능하도록)
  window.history.pushState({}, '', url);
}

function getURLParams() {
  const url = new URL(window.location);
  const params = {};
  if (url.searchParams.get('region')) params.region = url.searchParams.get('region');
  if (url.searchParams.get('dob')) params.dob = url.searchParams.get('dob');
  if (url.searchParams.get('category')) params.category = url.searchParams.get('category');
  if (url.searchParams.get('kw_text')) params.kw_text = url.searchParams.get('kw_text');
  return params;
}

function restoreFromURL() {
  const params = getURLParams();
  const currentPath = window.location.pathname;
  
  // /all 경로이거나 kw_text만 있으면 전체검색 모드
  const isGlobalMode = currentPath === '/all' || (!params.region && !params.dob && !params.category && params.kw_text);
  
  if (Object.keys(params).length === 0) return false;

  // 맞춤정보 모드인지 전체검색 모드인지 판단
  const hasPersonalParams = params.region || params.dob || params.category;
  
  if (hasPersonalParams && !isGlobalMode) {
    // 맞춤정보 모드 복원
    if (startCards) startCards.classList.add("hidden");
    personalForm.classList.add("hidden");
    globalSection.classList.add("hidden");
    customSummary.classList.remove("hidden");
    customSearch.classList.remove("hidden");
    
    // 요약 정보 복원
    const region = params.region || "전국";
    const cats = params.category ? params.category.split(",") : [];
    const dob = params.dob || "";
    const age = calculateAge(dob);
    
    document.getElementById("summaryName").textContent = "내 맞춤 정보";
    let summaryLine = `${region} | ${cats.join(", ")}`;
    if (age !== null) summaryLine += ` | ${age}세`;
    document.getElementById("summaryLine").textContent = summaryLine;
    
    // 폼 값 복원
    if (params.region) document.getElementById("regionSelect").value = params.region;
    if (params.dob) document.getElementById("dob").value = params.dob;
    if (params.category) {
      const catArray = params.category.split(",");
      tags.forEach(t => {
        if (catArray.includes(t.dataset.cat)) {
          t.classList.add("active");
        } else {
          t.classList.remove("active");
        }
      });
    }
    
    // 검색어 복원
    if (params.kw_text) {
      document.getElementById("kwTextCustom").value = params.kw_text;
    }
    
    currentParams = { ...params };
    // URL 복원 시에는 fetchResults에서 URL을 다시 업데이트하지 않도록 플래그 필요
    // 하지만 일단은 그대로 두고, fetchResults에서 중복 업데이트가 발생해도 문제없음
    fetchResults(currentParams);
    return true;
  } else if (params.kw_text || isGlobalMode) {
    // 전체검색 모드 복원 (/all 경로이거나 kw_text만 있는 경우)
    if (startCards) startCards.classList.add("hidden");
    globalSection.classList.remove("hidden");
    personalForm.classList.add("hidden");
    customSummary.classList.add("hidden");
    customSearch.classList.add("hidden");
    if (params.kw_text) {
      document.getElementById("kwText").value = params.kw_text;
    }
    currentParams = { ...params };
    fetchResults(currentParams);
    return true;
  }
  
  return false;
}

// ===== 서버 조회 =====
async function fetchResults(params) {
  loading.classList.remove("hidden");
  results.classList.add("hidden");
  results.innerHTML = "";
  loadMoreBtn.style.display = "none";

  // currentParams 업데이트
  currentParams = { ...currentParams, ...params };
  
  // URL 업데이트 (검색 결과 표시 시)
  updateURL(currentParams);

  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentParams)
    });
    if (!resp.ok) throw new Error(`서버 에러: ${resp.status}`);
    const data = await resp.json();

    // 서버 결과 저장
    lastResults = data;
    // 현재 화면(지원형태 필터 반영) 결과로 복사
    viewResults = [...lastResults];

    // 페이지 개수 초기화
    shownCount = Math.min(PAGE_SIZE, viewResults.length);

    renderResults(viewResults, { append: false });
  } catch (err) {
    results.innerHTML = `<p class="error">오류 발생: ${err.message}</p>`;
  } finally {
    loading.classList.add("hidden");
    results.classList.remove("hidden");
  }
}
