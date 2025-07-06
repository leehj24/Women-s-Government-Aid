// 엘리먼트 참조
const personalCard        = document.getElementById('personalCard');
const globalCard          = document.getElementById('globalCard');
const personalForm        = document.getElementById('personalForm');
const globalSection       = document.getElementById('globalSection');
const filterForm          = document.getElementById('filterForm');
const supportSelect       = document.getElementById('supportSelect');
const btnFilterToggle     = document.getElementById('btnFilterToggle');
const btnFilterSearch     = document.getElementById('btnFilterSearch');
const customSummary       = document.getElementById('customSummary');
const btnEditInfo         = document.getElementById('btnEditInfo');
const customSearch        = document.getElementById('customSearch');
const btnSearch           = document.getElementById('btnSearch');
const btnSearchText       = document.getElementById('btnSearchText');
const btnSearchTextCustom = document.getElementById('btnSearchTextCustom');
const results             = document.getElementById('results');
const loading             = document.getElementById('loading');
const tags                = document.querySelectorAll('.tag');

// 전역 결과 저장용
let lastResults = [];
let currentParams = {};

// helper: 나이 계산
function calculateAge(dob) {
  if (!dob) return null;
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (today.getMonth() < birth.getMonth() ||
      (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())) {
    age--;
  }
  return age;
}

// 초기화: 모든 폼/결과 숨기기
[personalForm, globalSection, filterForm, customSummary, customSearch, results]
  .forEach(el => el.classList.add('hidden'));

// 시작 카드 이벤트
personalCard.onclick = () => {
  document.querySelector('.start-cards').classList.add('hidden');
  personalForm.classList.remove('hidden');
};

globalCard.onclick = () => {
  document.querySelector('.start-cards').classList.add('hidden');
  globalSection.classList.remove('hidden');
  filterForm.classList.add('hidden');
  customSummary.classList.add('hidden');
  customSearch.classList.add('hidden');
  fetchResults({});
};

// 조건검색 토글
btnFilterToggle.onclick = () => filterForm.classList.toggle('hidden');

// 맞춤 완료: 지역·카테고리·나이 포함
btnSearch.onclick = () => {
  const region = document.getElementById('regionSelect').value;
  const dob    = document.getElementById('dob').value;
  const cats   = Array.from(tags)
    .filter(t => t.classList.contains('active'))
    .map(t => t.dataset.cat);
  const age = calculateAge(dob);

  currentParams = { region, dob, category: cats.join(',') };
  document.getElementById('summaryName').textContent = '내 맞춤 정보';
  let summaryLine = `${region} | ${cats.join(', ')}`;
  if (age !== null) summaryLine += ` | ${age}세`;
  document.getElementById('summaryLine').textContent = summaryLine;

  personalForm.classList.add('hidden');
  customSummary.classList.remove('hidden');
  customSearch.classList.remove('hidden');
  fetchResults(currentParams);
};

// 정보수정
btnEditInfo.onclick = () => {
  customSummary.classList.add('hidden');
  customSearch.classList.add('hidden');
  results.innerHTML = '';
  personalForm.classList.remove('hidden');
};

// 검색 🔍 (전체·맞춤)
btnSearchText.onclick = () => {
  const kw = document.getElementById('kwText').value.trim();
  fetchResults({ ...currentParams, kw_text: kw });
};

btnSearchTextCustom.onclick = () => {
  const kw = document.getElementById('kwTextCustom').value.trim();
  fetchResults({ ...currentParams, kw_text: kw });
};

// 태그 토글
tags.forEach(t => t.addEventListener('click', () => t.classList.toggle('active')));

// 지원형태 클릭 시: 클라이언트 필터링
btnFilterSearch.onclick = () => {
  const supArr = Array.from(supportSelect.selectedOptions).map(o => o.value);
  if (!supArr.length) {
    renderResults(lastResults);
    return;
  }
  const filtered = lastResults.filter(item => {
    const sups = (item['지원형태_분류'] || '').split(',').map(x => x.trim());
    return supArr.some(s => sups.includes(s));
  });
  renderResults(filtered);
};

// 결과 렌더링 함수 분리
function renderResults(data) {
  if (!data.length) {
    results.innerHTML = '<p>검색 결과가 없습니다.</p>';
    return;
  }
  results.innerHTML = data.map(item => `
    <div class="card result-card" onclick="location.href='/detail/${item.index}'">
      <h3>${item.제목}</h3>
      <p>${item.카테고리_분류}</p>
      <p>${item.지역}</p>
    </div>
  `).join('');
}

// fetch + 저장 + 렌더링 + 로딩 제어
async function fetchResults(params) {
  loading.classList.remove('hidden');
  results.classList.add('hidden');
  results.innerHTML = '';
  try {
    const resp = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!resp.ok) throw new Error(`서버 에러: ${resp.status}`);
    const data = await resp.json();
    lastResults = data;  // 결과 저장
    renderResults(data);
  } catch (err) {
    results.innerHTML = `<p class="error">오류 발생: ${err.message}</p>`;
  } finally {
    loading.classList.add('hidden');
    results.classList.remove('hidden');
  }
}
