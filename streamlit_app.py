# -*- coding: utf-8 -*-
import os, json, re, time, uuid
import streamlit as st
import pandas as pd
import numpy as np

# 지도
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
from streamlit_folium import st_folium

st.set_page_config(page_title="디지털 노마드 지역 추천 대시보드", layout="wide")

# ------------------------------- 경로/폴더 -------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CANDIDATE_BASES = [DATA_DIR, APP_DIR, "."] # 실행 파일 기준 폴더 추가

def build_paths():
    for base in CANDIDATE_BASES:
        fv = os.path.join(base, "20250809144224_광역별 방문자 수.csv")
        fc = os.path.join(base, "PLP_업종별_검색건수_통합.csv")
        ft = os.path.join(base, "PLP_유형별_검색건수_통합.csv")
        # 모든 파일이 존재하는지 확인하는 대신, 각 파일 경로를 반환하도록 수정
        # 파일 존재 여부는 로딩 함수에서 처리
    return (
        os.path.join(CANDIDATE_BASES[0], "20250809144224_광역별 방문자 수.csv"),
        os.path.join(CANDIDATE_BASES[0], "PLP_업종별_검색건수_통합.csv"),
        os.path.join(CANDIDATE_BASES[0], "PLP_유형별_검색건수_통합.csv"),
    )

file_visitors, file_search_cat, file_search_type = build_paths()

def resolve_geojson_path():
    env_path = os.environ.get("GEOJSON_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path
    for p in [
        os.path.join(DATA_DIR, "korea_provinces.geojson"),
        os.path.join(DATA_DIR, "KOREA_GEOJSON.geojson"),
        os.path.join(APP_DIR,  "korea_provinces.geojson"),
        os.path.join(APP_DIR,  "KOREA_GEOJSON.geojson"),
    ]:
        if os.path.exists(p):
            return p
    return ""

KOREA_GEOJSON = resolve_geojson_path()
GEO_PROP_KEYS = ["name", "CTPRVN_NM", "ADM1_KOR_NM", "sido_nm", "SIG_KOR_NM", "NAME_1"]

# ---- 인프라 폴더/ZIP 자동 탐색 ----
INFRA_DIR_NAME  = "소상공인시장진흥공단_상가(상권)정보_20250630"
INFRA_ZIP_NAME  = "소상공인시장진흥공단_상가(상권)정보_20250630.zip"

def resolve_infra_sources():
    for base in CANDIDATE_BASES:
        d = os.path.join(base, INFRA_DIR_NAME)
        if os.path.isdir(d):
            csvs = []
            for r, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(".csv"):
                        csvs.append(os.path.join(r, f))
            if csvs:
                return {"mode": "dir", "paths": csvs}
    for base in CANDIDATE_BASES:
        z = os.path.join(base, INFRA_ZIP_NAME)
        if os.path.exists(z):
            return {"mode": "zip", "paths": [z]}
    return {"mode": "none", "paths": []}

# ---------------------------- 표준/약식 지역명 ----------------------------
TWOCHAR_MAP = {"서울":"서울","부산":"부산","대구":"대구","인천":"인천","광주":"광주","대전":"대전","울산":"울산","세종":"세종",
               "경기":"경기","강원":"강원","충북":"충북","충남":"충남","전북":"전북","전남":"전남","경북":"경북","경남":"경남","제주":"제주"}

def to_twochar(s: str) -> str:
    s = str(s)
    if s.startswith("전라남"): return "전남"
    if s.startswith("전라북"): return "전북"
    if s.startswith("경상남"): return "경남"
    if s.startswith("경상북"): return "경북"
    if s.startswith("충청남"): return "충남"
    if s.startswith("충청북"): return "충북"
    s = re.sub(r"(특별자치도|특별자치시|특별시|광역시|자치도|자치시|도|시)$","", s)
    return TWOCHAR_MAP.get(s, s[:2])

REGION_COORDS = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756), "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052), "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890), "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555), "충북": (36.6357, 127.4913), "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530), "전남": (34.8679, 126.9910), "경북": (36.4919, 128.8889),
    "경남": (35.4606, 128.2132), "제주": (33.4996, 126.5312),
}

def normalize_region_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\(.*?\)", "", str(s))
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s).strip()
    for a in ["특별자치도","특별자치시","특별시","광역시","자치도","자치시","도","시"]:
        s = s.replace(a, "")
    s = re.sub(r"\s+", " ", s).strip()
    return to_twochar(s)

def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0: return s.fillna(0.0)
    d = s.max() - s.min()
    return (s - s.min())/d if d>0 else s*0

# --------------------------- 안전 GeoJSON 로더 ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson_safe(path: str):
    if not path or not os.path.exists(path):
        return None, "missing_path"
    try:
        if os.path.getsize(path) == 0:
            return None, "empty_file"
    except Exception:
        pass
    for enc in ("utf-8","utf-8-sig","utf-16","utf-16-le","utf-16-be","cp949","euc-kr","latin-1"):
        try:
            with open(path,"r",encoding=enc,errors="strict") as f:
                txt=f.read().strip()
            if not txt: return None, f"empty_text({enc})"
            if txt.lstrip().startswith(("var ","const ","let ")):
                m=re.search(r"\{.*\}\s*;?\s*$", txt, flags=re.S)
                if m: txt=m.group(0)
            gj=json.loads(txt)
            if not isinstance(gj,dict): continue
            if gj.get("type")=="Topology": return None, "topojson_not_supported"
            if gj.get("type")!="FeatureCollection" or not isinstance(gj.get("features"),list):
                continue
            return gj, None
        except Exception:
            continue
    return None, "parse_error"

# ----------------------------- CSV 로더 -----------------------------
NEEDED_VIS_COLS = ["광역지자체명", "기초지자체 방문자 수"]

@st.cache_data(show_spinner=False)
def read_csv_forgiving(path, usecols=None, dtype=None):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, dtype=dtype, low_memory=False)
        except Exception:
            continue
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def read_visitors():
    return read_csv_forgiving(file_visitors, usecols=NEEDED_VIS_COLS, dtype={"광역지자체명":"string"})

# ----- (그래프용) 검색건수 파일 로더: 유연 컬럼 감지 -----
@st.cache_data(show_spinner=False)
def load_search_counts(path):
    df = read_csv_forgiving(path)
    if df.empty: return pd.DataFrame(), (None, None, None)

    cols={c.lower():c for c in df.columns}
    rcol=None
    for k in ["지역","시도","시도명","광역지자체","sido","region","province"]:
        if k in cols: rcol=cols[k]; break
    gcol=None
    for k in ["중분류","대분류","업종","카테고리","유형","키워드","검색어","항목"]:
        if k in cols: gcol=cols[k]; break
    vcol=None
    for k in ["검색건수","검색 건수","검색수","검색량","count","건수","value","합계","총건수"]:
        if k in cols: vcol=cols[k]; break
    if vcol: df[vcol]=pd.to_numeric(df[vcol], errors="coerce")
    return df, (rcol,gcol,vcol)

# ----------------------- 핵심 지표 계산 -----------------------
vis = read_visitors()

# --- 오류 수정 부분 시작 ---
# 방문자 파일이 없을 경우를 대비한 안정성 강화
vis_region = pd.DataFrame()
if not vis.empty:
    vis_region = (vis.groupby("광역지자체명", as_index=False)["기초지자체 방문자 수"]
                   .sum()
                   .rename(columns={"기초지자체 방문자 수":"방문자수_합계"}))
    total_visitors = max(vis_region["방문자수_합계"].sum(), 1)
    vis_region["방문자_점유율"] = vis_region["방문자수_합계"] / total_visitors
    vis_region["지역_norm"] = vis_region["광역지자체명"].map(normalize_region_name)
else:
    # st.warning을 사용하여 앱 화면에 경고 메시지 표시
    st.warning(f"방문자 수 데이터 파일({os.path.basename(file_visitors)})을 찾을 수 없습니다. 방문자 관련 점수는 0으로 처리됩니다.")
    all_regions = list(REGION_COORDS.keys())
    vis_region = pd.DataFrame({
        "광역지자체명": all_regions,
        "방문자수_합계": 0,
        "방문자_점유율": 0,
        "지역_norm": [normalize_region_name(r) for r in all_regions]
    })
# --- 오류 수정 부분 끝 ---

# 기본 메트릭: 방문자만 반영
metrics_map = vis_region.copy()
coords_df = pd.DataFrame([{"지역_norm":k,"lat":v[0],"lon":v[1]} for k,v in REGION_COORDS.items()])
metrics_map = metrics_map.merge(coords_df, on="지역_norm", how="left")
metrics_map["방문자_점유율_norm"] = minmax(metrics_map["방문자_점유율"].fillna(0))
metrics_map["NSI_base"] = metrics_map["방문자_점유율_norm"].fillna(0)

# ==================== 인프라 지표(상가 폴더/ZIP) 통합 ====================
@st.cache_data(show_spinner=True)
def build_infra_from_sources(sources):
    import io, zipfile
    required = {"시도명","상권업종중분류명","상권업종소분류명","표준산업분류명"}
    dfs = []
    if sources["mode"] == "dir":
        for path in sources["paths"]:
            df=read_csv_forgiving(path)
            if df.empty or not required.issubset(set(df.columns)):
                continue
            dfs.append(df[list(required)].copy())
    elif sources["mode"] == "zip":
        zpath = sources["paths"][0]
        with zipfile.ZipFile(zpath,"r") as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"): continue
                raw=z.read(name)
                df=read_csv_forgiving(io.BytesIO(raw))
                if df.empty or not required.issubset(set(df.columns)):
                    continue
                dfs.append(df[list(required)].copy())
    if not dfs: return pd.DataFrame()

    df=pd.concat(dfs, ignore_index=True)
    for c in required: df[c]=df[c].astype(str).str.strip()

    mid=df["상권업종중분류명"].astype(str)
    sub=df["상권업종소분류명"].astype(str)
    std=df["표준산업분류명"].astype(str)

    m_cafe      = (sub.str.contains("카페")) | (std.str.contains("커피 전문점"))
    m_conv      = (sub.str.contains("편의점")) | (std.str.contains("체인화 편의점"))
    m_laundry   = sub.str.contains("세탁소|빨래방")
    m_pharmacy  = sub.str.contains("약국")
    m_clinic    = mid.str.contains("의원")
    m_hospital  = mid.str.contains("병원") | m_clinic | sub.str.contains("치과의원|한의원")
    m_library   = mid.str.contains("도서관|사적지")
    m_pc        = sub.str.contains("PC방")
    m_karaoke   = sub.str.contains("노래방|노래연습장")
    m_fitness   = sub.str.contains("헬스장")
    m_yoga      = sub.str.contains("요가|필라테스")

    df["sido_norm"]=df["시도명"].map(normalize_region_name)
    agg=df.groupby("sido_norm").agg(
        total_places=("시도명","size"),
        cafe_count=("시도명",             lambda s:int(m_cafe.loc[s.index].sum())),
        convenience_count=("시도명",      lambda s:int(m_conv.loc[s.index].sum())),
        laundry_count=("시도명",          lambda s:int(m_laundry.loc[s.index].sum())),
        pharmacy_count=("시도명",         lambda s:int(m_pharmacy.loc[s.index].sum())),
        hospital_count=("시도명",         lambda s:int(m_hospital.loc[s.index].sum())),
        library_museum_count=("시도명",   lambda s:int(m_library.loc[s.index].sum())),
        pc_cafe_count=("시도명",          lambda s:int(m_pc.loc[s.index].sum())),
        karaoke_count=("시도명",          lambda s:int(m_karaoke.loc[s.index].sum())),
        fitness_count=("시도명",          lambda s:int(m_fitness.loc[s.index].sum())),
        yoga_pilates_count=("시도명",     lambda s:int(m_yoga.loc[s.index].sum())),
    ).reset_index()

    def per_10k(n,total): total=total.replace(0,np.nan); return (n/total)*10000

    metric_cols = [
        "cafe_count","convenience_count","laundry_count","pharmacy_count",
        "hospital_count","library_museum_count","pc_cafe_count",
        "karaoke_count","fitness_count","yoga_pilates_count"
    ]
    for col in metric_cols:
        agg[col+"_per10k"] = per_10k(agg[col].astype(float), agg["total_places"].astype(float)).round(3)
        v=agg[col].astype(float); rng=v.max()-v.min()
        agg[col+"_norm"] = ((v-v.min())/rng).fillna(0).round(4) if rng>0 else v*0
    return agg

# ==================== KTX 데이터 로드 ====================
def find_optional_file(names):
    for base in CANDIDATE_BASES:
        for n in names:
            p=os.path.join(base,n)
            if os.path.exists(p): return p
    return ""

@st.cache_data(show_spinner=False)
def load_ktx_counts(path):
    df = read_csv_forgiving(path)
    if df.empty: return pd.DataFrame()

    cols={c.lower():c for c in df.columns}
    sido_col=cols.get("시도") or cols.get("시도명") or cols.get("광역지자체")
    if not sido_col:
        addr_col=None
        addr_keys = ["역주소","주소","소재지","소재지주소","역사주소","지번주소","도로명주소","역사 도로명주소"]
        for key in addr_keys:
            if key in cols: addr_col = cols[key]; break
        if not addr_col:
            for key in addr_keys:
                 for c in df.columns:
                    if key in c: addr_col=c; break
                 if addr_col: break
        if not addr_col: return pd.DataFrame()

        sidos=df[addr_col].astype(str)
        def ext(addr):
            m=re.match(r"(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|제주특별자치도|경기도|강원특별자치도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도)", addr.strip())
            return m.group(1) if m else ""
        sidos=sidos.map(ext)
    else:
        sidos=df[sido_col].astype(str)
    ktx=pd.DataFrame({"지역_norm":sidos.map(normalize_region_name)})
    ktx=ktx[ktx["지역_norm"].astype(str).str.len()>0]
    return ktx.value_counts("지역_norm").rename("ktx_cnt").reset_index()

@st.cache_data(show_spinner=False)
def load_coworking(path):
    df = read_csv_forgiving(path)
    if df.empty: return pd.DataFrame()

    cols_lower={c.lower():c for c in df.columns}
    sido_col=None
    for k in ["시도","시도명","광역지자체","sido"]:
        if k in df.columns: sido_col=k; break
        if k in cols_lower: sido_col=cols_lower[k]; break
    if not sido_col:
        addr_col=None
        for key in ["주소","소재지","도로명주소","상세주소","지번주소"]:
            for c in df.columns:
                if key in c: addr_col=c; break
            if addr_col: break
        if not addr_col: return pd.DataFrame()
        sidos=df[addr_col].astype(str)
        def ext(addr):
            m=re.match(r"(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|제주특별자치도|경기도|강원특별자치도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도)", addr.strip())
            return m.group(1) if m else ""
        sidos=sidos.map(ext)
    else:
        sidos=df[sido_col].astype(str)
    g=pd.DataFrame({"지역_norm":sidos.map(normalize_region_name)})
    g=g[g["지역_norm"]!=""]
    return g.value_counts("지역_norm").rename("coworking_sites").reset_index()

# ============================ UI 레이아웃 ============================
st.title("디지털 노마드 지역 추천 대시보드")
left, right = st.columns([2, 1])
with left:
    st.subheader("지도에서 지역을 선택하세요")

# -------- 사이드바 --------
st.sidebar.header("추천 카테고리")
CATEGORY_OPTIONS = [
    "🔥 현재 인기 지역",
    "🚉 교통 좋은 지역",
    "🏛 코워킹 인프라 풍부 지역",
    "💰 합리적인인 비용",
    "🚀 빠른 인터넷",
]
selected_category = st.sidebar.selectbox("하나만 선택", CATEGORY_OPTIONS, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("주변 인프라 선택 (보너스 점수)")
cb_infra_cafe    = st.sidebar.checkbox("☕ 카페", value=False)
cb_infra_lib     = st.sidebar.checkbox("🏛️ 도서관", value=False)
cb_infra_conv    = st.sidebar.checkbox("🏪 편의점", value=False)
cb_infra_laundry = st.sidebar.checkbox("🧺 세탁소", value=False)
cb_infra_hos     = st.sidebar.checkbox("🏥 병·의원", value=False)
cb_infra_phar    = st.sidebar.checkbox("💊 약국", value=False)
cb_infra_pc      = st.sidebar.checkbox("💻 PC방", value=False)
cb_infra_karaoke = st.sidebar.checkbox("🎤 노래방", value=False)
cb_infra_fit     = st.sidebar.checkbox("💪 헬스장", value=False)
cb_infra_yoga    = st.sidebar.checkbox("🧘 요가/필라테스", value=False)

# ----------------- 데이터 로드/계산 -----------------
need_infra = any([
    cb_infra_cafe, cb_infra_lib, cb_infra_conv, cb_infra_laundry,
    cb_infra_hos, cb_infra_phar, cb_infra_pc, cb_infra_karaoke,
    cb_infra_fit, cb_infra_yoga, selected_category == "🏛 코워킹 인프라 풍부 지역"
])
need_access = (selected_category=="🚉 교통 좋은 지역")
need_cowork = (selected_category=="🏛 코워킹 인프라 풍부 지역")

if need_infra:
    infra_sources = resolve_infra_sources()
    infra_df = build_infra_from_sources(infra_sources)
    if not infra_df.empty:
        metrics_map = metrics_map.merge(
            infra_df.add_prefix("infra__").rename(columns={"infra__sido_norm":"지역_norm"}),
            on="지역_norm", how="left"
        )
else:
    infra_df = pd.DataFrame()

if need_access:
    ktx_file_path = find_optional_file([
        "한국철도공사_KTX 노선별 역정보_20240411.csv", "KTX_노선별_역정보.csv", "ktx_stations.csv"
    ])
    ktx_df = load_ktx_counts(ktx_file_path)

    if not ktx_df.empty:
        metrics_map = metrics_map.merge(ktx_df, on="지역_norm", how="left")
        metrics_map["ktx_cnt"] = metrics_map["ktx_cnt"].fillna(0)
        metrics_map["access_score"] = minmax(metrics_map["ktx_cnt"])
    else:
        st.warning(f"KTX 역 정보 파일({os.path.basename(ktx_file_path)})을 찾을 수 없습니다.")
        metrics_map["ktx_cnt"] = 0
        metrics_map["access_score"] = 0.0

if need_cowork:
    cowork_files = [
        "KC_CNRS_OFFM_FCLTY_DATA_2023.csv","공유오피스.csv","coworking_sites.csv",
        "중소벤처기업진흥공단_공유오피스_운영현황.csv","한국문화정보원_전국공유오피스시설데이터.csv","전국_공유_오피스_시설_데이터.csv"
    ]
    cw_file = find_optional_file(cowork_files)
    cow_df = load_coworking(cw_file)
    if not cow_df.empty:
        if need_infra and "infra__total_places" in metrics_map.columns:
            base = metrics_map[["지역_norm","infra__total_places"]]
            cow = cow_df.merge(base, on="지역_norm", how="left")
            cow["cowork_per10k"] = (cow["coworking_sites"]/cow["infra__total_places"].replace(0,np.nan)*10000).round(3)
            v = pd.to_numeric(cow["cowork_per10k"], errors="coerce")
        else:
            cow = cow_df.copy()
            cow["cowork_per10k"]=np.nan
            v = pd.to_numeric(cow["coworking_sites"], errors="coerce")
        rng=(v.max()-v.min())
        cow["cowork_norm"]=((v-v.min())/rng).fillna(0) if rng>0 else (v*0)
        metrics_map = metrics_map.merge(cow[["지역_norm","coworking_sites","cowork_per10k","cowork_norm"]], on="지역_norm", how="left")

# ----------------------------- 룰/보너스 -----------------------------
def _compute_bonus_columns(g, selected_category):
    CAT_BONUS   = 0.15
    INFRA_BONUS = 0.10
    q_vis_hi = g["방문자_점유율_norm"].quantile(0.70)
    q_acc_hi = g["access_score"].quantile(0.70) if "access_score" in g and g["access_score"].notna().any() else 1.0
    q_cwk_hi = g["cowork_norm"].quantile(0.70) if "cowork_norm" in g and g["cowork_norm"].notna().any() else 1.0

    bonus = np.zeros(len(g), dtype=float)

    def add_above(series, q):
        s=pd.to_numeric(series, errors="coerce").fillna(0)
        return (s - q).clip(lower=0)

    if selected_category == "🔥 현재 인기 지역":
        bonus += CAT_BONUS * add_above(g["방문자_점유율_norm"], q_vis_hi)
    elif selected_category == "🚉 교통 좋은 지역" and "access_score" in g:
        bonus += CAT_BONUS * add_above(g["access_score"], q_acc_hi)
    elif selected_category == "🏛 코워킹 인프라 풍부 지역" and "cowork_norm" in g:
        bonus += CAT_BONUS * add_above(g["cowork_norm"], q_cwk_hi)
    elif selected_category == "💰 합리적인인 비용" and "cost_index" in g and g["cost_index"].notna().any():
        cost = g["cost_index"]
        rng = cost.max() - cost.min()
        if rng > 0:
            normalized_cost = (cost - cost.min()) / rng
            bonus += CAT_BONUS * (1 - normalized_cost)
    elif selected_category == "🚀 빠른 인터넷" and "internet_mbps" in g and g["internet_mbps"].notna().any():
         bonus += CAT_BONUS * add_above(g["internet_mbps"], g["internet_mbps"].quantile(0.70))

    if cb_infra_cafe and "infra__cafe_count_norm" in g: bonus += INFRA_BONUS * g["infra__cafe_count_norm"].fillna(0)
    if cb_infra_lib and "infra__library_museum_count_norm" in g: bonus += INFRA_BONUS * g["infra__library_museum_count_norm"].fillna(0)
    if cb_infra_conv and "infra__convenience_count_norm" in g: bonus += INFRA_BONUS * g["infra__convenience_count_norm"].fillna(0)
    if cb_infra_laundry and "infra__laundry_count_norm" in g: bonus += INFRA_BONUS * g["infra__laundry_count_norm"].fillna(0)
    if cb_infra_hos and "infra__hospital_count_norm" in g: bonus += INFRA_BONUS * g["infra__hospital_count_norm"].fillna(0)
    if cb_infra_phar and "infra__pharmacy_count_norm" in g: bonus += INFRA_BONUS * g["infra__pharmacy_count_norm"].fillna(0)
    if cb_infra_pc and "infra__pc_cafe_count_norm" in g: bonus += INFRA_BONUS * g["infra__pc_cafe_count_norm"].fillna(0)
    if cb_infra_karaoke and "infra__karaoke_count_norm" in g: bonus += INFRA_BONUS * g["infra__karaoke_count_norm"].fillna(0)
    if cb_infra_fit and "infra__fitness_count_norm" in g: bonus += INFRA_BONUS * g["infra__fitness_count_norm"].fillna(0)
    if cb_infra_yoga and "infra__yoga_pilates_count_norm" in g: bonus += INFRA_BONUS * g["infra__yoga_pilates_count_norm"].fillna(0)
    return bonus

def apply_category_rules_all(g):
    g = g.copy()
    bonus = _compute_bonus_columns(g, selected_category)
    g["NSI"] = (g["NSI_base"] + bonus).clip(0,1)
    return g

def apply_category_rules(g):
    g = g.copy()
    mask = pd.Series(True, index=g.index)
    if selected_category == "🔥 현재 인기 지역":
        mask = (g["방문자_점유율_norm"] >= g["방문자_점유율_norm"].quantile(0.70))
    elif selected_category == "🚉 교통 좋은 지역" and "access_score" in g and g["access_score"].notna().any() and g["access_score"].sum() > 0:
        mask = (g["access_score"] >= g["access_score"].quantile(0.70))
    elif selected_category == "🏛 코워킹 인프라 풍부 지역" and "cowork_norm" in g and g["cowork_norm"].notna().any() and g["cowork_norm"].sum() > 0:
        mask = (g["cowork_norm"] >= g["cowork_norm"].quantile(0.70))
    elif selected_category == "💰 합리적인인 비용" and "cost_index" in g and g["cost_index"].notna().any():
        mask = (g["cost_index"] <= g["cost_index"].quantile(0.30))
    elif selected_category == "🚀 빠른 인터넷" and "internet_mbps" in g and g["internet_mbps"].notna().any():
        mask = (g["internet_mbps"] >= g["internet_mbps"].quantile(0.70))

    g = g.loc[mask].copy()
    bonus = _compute_bonus_columns(g, selected_category)
    g["NSI"] = (g["NSI_base"] + bonus).clip(0,1)
    return g

metrics_all = apply_category_rules_all(metrics_map)
metrics_after_rules = apply_category_rules(metrics_map)

# ----------------------------- 랭킹 계산 -----------------------------
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
selected_norm = normalize_region_name(st.session_state.selected_region) if st.session_state.selected_region else None

ranked_all = metrics_all.copy()
ranked_all["NSI"]  = ranked_all["NSI"].fillna(ranked_all["NSI_base"]).fillna(0.0).clip(0,1)
ranked_all["rank"] = ranked_all["NSI"].rank(ascending=False, method="min").astype(int)

ranked = metrics_after_rules.copy()
if not ranked.empty:
    ranked["NSI"]  = ranked["NSI"].fillna(ranked["NSI_base"]).fillna(0.0).clip(0,1)
    ranked["rank"] = ranked["NSI"].rank(ascending=False, method="min").astype(int)
else:
    ranked["rank"] = 0

COLOR_TOP1, COLOR_TOP2, COLOR_TOP3 = "#e60049", "#ffd43b", "#4dabf7"
COLOR_SEL, COLOR_BASE = "#51cf66", "#cfd4da"
def pick_color(region_norm, selected_region_norm=None):
    if selected_region_norm and region_norm == selected_region_norm:
        return COLOR_SEL
    if not ranked.empty:
        r_series = ranked.loc[ranked["지역_norm"]==region_norm, "rank"]
        r = int(r_series.min()) if not r_series.empty else 999
        return {1:COLOR_TOP1, 2:COLOR_TOP2, 3:COLOR_TOP3}.get(r, COLOR_BASE)
    return COLOR_BASE

# =============================== 지도 ===============================
MAP_HEIGHT = 680
with left:
    m = folium.Map(location=[36.5,127.8], zoom_start=7, tiles="cartodbpositron", prefer_canvas=True)
    gj, gj_err = load_geojson_safe(KOREA_GEOJSON)

    if gj:
        # GeoJSON에 랭킹 정보 추가
        # (이하 코드 동일)
        pass # The rest of the map rendering code is unchanged

    map_state = st_folium(m, width=None, height=MAP_HEIGHT, key="main_map")
    # (이하 코드 동일)
    pass

# ============================ 우측 패널 ============================
with right:
    # (이하 코드 동일)
    pass

# ============================ 랭킹/다운로드 ============================
st.subheader("추천 랭킹")
# (이하 코드 동일)
pass

# ============================ 키워드 · 카테고리 탐색 ============================
st.markdown("## 키워드 · 카테고리 탐색")
# (이하 코드 동일)
pass

# ============================ 출처 ============================
st.markdown("""
---
**데이터 출처**
- 한국관광데이터랩: 지역별 방문자수, 지역별 검색건수
- 소상공인시장진흥공단: 상가(상권) 정보
- **한국철도공사**: KTX 노선별 역정보
- 한국문화정보원: 전국공유오피스시설데이터
""")

# (전체 코드를 붙여넣기 위해 생략된 부분들은 이전 코드와 동일합니다.)
# 위 코드의 생략된 부분들을 채워서 전체 코드를 완성할 수 있습니다.
# ... [이전 답변의 지도, 우측패널, 랭킹, 키워드탐색 코드 전체를 여기에 붙여넣으세요] ...
# Due to length constraints, the unchanged parts of the UI (map, right panel, ranking table) are omitted here.
# Please paste the corresponding sections from the previous complete code block to finalize the script.
