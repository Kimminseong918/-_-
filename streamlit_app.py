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
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CANDIDATE_BASES = [DATA_DIR, APP_DIR, "/mnt/data"]

def build_paths():
    for base in CANDIDATE_BASES:
        fv = os.path.join(base, "20250809144224_광역별 방문자 수.csv")
        fc = os.path.join(base, "PLP_업종별_검색건수_통합.csv")
        ft = os.path.join(base, "PLP_유형별_검색건수_통합.csv")
        if all(os.path.exists(p) for p in [fv, fc, ft]):
            return fv, fc, ft
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
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, dtype=dtype, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)

@st.cache_data(show_spinner=False)
def read_visitors():
    return read_csv_forgiving(file_visitors, usecols=NEEDED_VIS_COLS, dtype={"광역지자체명":"string"})

# ----- (그래프용) 검색건수 파일 로더: 유연 컬럼 감지 -----
@st.cache_data(show_spinner=False)
def load_search_counts(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(), (None, None, None)
    df=None
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try:
            df=pd.read_csv(path, encoding=enc, low_memory=False); break
        except Exception:
            df=None
    if df is None or df.empty: return pd.DataFrame(), (None, None, None)
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

# 방문자 집계/정규화
vis_region = (vis.groupby("광역지자체명", as_index=False)["기초지자체 방문자 수"]
                .sum()
                .rename(columns={"기초지자체 방문자 수":"방문자수_합계"}))
total_visitors = max(vis_region["방문자수_합계"].sum(), 1)
vis_region["방문자_점유율"] = vis_region["방문자수_합계"] / total_visitors
vis_region["지역_norm"] = vis_region["광역지자체명"].map(normalize_region_name)

# 기본 메트릭(숙박 제외): 방문자만 반영
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
            df=None
            for enc in ["cp949","utf-8","euc-kr","latin1"]:
                try:
                    df=pd.read_csv(path, encoding=enc, low_memory=False); break
                except Exception:
                    df=None
            if df is None or not required.issubset(set(df.columns)): 
                continue
            dfs.append(df[list(required)].copy())
    elif sources["mode"] == "zip":
        zpath = sources["paths"][0]
        with zipfile.ZipFile(zpath,"r") as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"): continue
                raw=z.read(name); df=None
                for enc in ["cp949","utf-8","euc-kr","latin1"]:
                    try:
                        df=pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False); break
                    except Exception:
                        df=None
                if df is None or not required.issubset(set(df.columns)): 
                    continue
                dfs.append(df[list(required)].copy())
    if not dfs: return pd.DataFrame()

    df=pd.concat(dfs, ignore_index=True)
    for c in required: df[c]=df[c].astype(str).str.strip()

    mid=df["상권업종중분류명"].astype(str)
    sub=df["상권업종소분류명"].astype(str)
    std=df["표준산업분류명"].astype(str)

    # ---- 최종 분류(키워드) ----
    m_cafe      = (sub.str.contains("카페")) | (std.str.contains("커피 전문점"))
    m_conv      = (sub.str.contains("편의점")) | (std.str.contains("체인화 편의점"))
    m_laundry   = sub.str.contains("세탁소|빨래방")
    m_pharmacy  = sub.str.contains("약국")
    m_clinic    = mid.str.contains("의원")
    m_hospital  = mid.str.contains("병원") | m_clinic | sub.str.contains("치과의원|한의원")
    m_library   = mid.str_contains("도서관·사적지") if hasattr(mid, "str_contains") else mid.str.contains("도서관·사적지")
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

# ==================== 교통 접근성 + KTX ====================
def find_optional_file(names):
    for base in CANDIDATE_BASES:
        for n in names:
            p=os.path.join(base,n)
            if os.path.exists(p): return p
    return ""

@st.cache_data(show_spinner=False)
def load_transport(path):
    if not path: return pd.DataFrame()
    df=None
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try: df=pd.read_csv(path, encoding=enc); break
        except Exception: df=None
    if df is None: return pd.DataFrame()
    cols={c.lower():c for c in df.columns}
    sidocol=cols.get("sido") or cols.get("시도") or cols.get("시도명")
    if not sidocol: return pd.DataFrame()
    df["_sido_"]=df[sidocol].astype(str).map(normalize_region_name)
    def pick(*names):
        for n in names:
            if n in df.columns: return n
        for n in names:
            if n in cols: return cols[n]
        return None
    ac=pick("airport_cnt","공항수","공항_cnt")
    kc=pick("ktx_cnt","ktx수","ktx_cnt")
    bc=pick("bus_term_cnt","버스터미널수","버스터미널_cnt")
    md=pick("min_dist_airport","최소공항거리_km","최근접공항거리")
    for c in [ac,kc,bc,md]:
        if c and c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
    parts=[]
    if ac: parts.append(df[ac].rank(pct=True))
    if kc: parts.append(df[kc].rank(pct=True))
    if bc: parts.append(df[bc].rank(pct=True))
    if md: parts.append(1-df[md].rank(pct=True))
    if not parts: return pd.DataFrame()
    access=sum(parts)/len(parts)
    out=pd.DataFrame({"지역_norm":df["_sido_"], "airport_cnt":df[ac] if ac else np.nan,
                      "ktx_cnt":df[kc] if kc else np.nan, "bus_term_cnt":df[bc] if bc else np.nan,
                      "min_dist_airport":df[md] if md else np.nan, "access_score":access})
    return out.groupby("지역_norm",as_index=False).mean()

@st.cache_data(show_spinner=False)
def load_ktx_counts(path):
    if not path or not os.path.exists(path): return pd.DataFrame()
    df=None
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try: df=pd.read_csv(path, encoding=enc, low_memory=False); break
        except Exception: df=None
    if df is None or df.empty: return pd.DataFrame()
    cols={c.lower():c for c in df.columns}
    sido_col=cols.get("시도") or cols.get("시도명") or cols.get("광역지자체")
    if not sido_col:
        addr_col=None
        for key in ["주소","소재지","소재지주소","역주소","역사주소","지번주소","도로명주소","역사 도로명주소"]:
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
    if not path: return pd.DataFrame()
    df=None
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try: df=pd.read_csv(path, encoding=enc, low_memory=False); break
        except Exception: df=None
    if df is None or df.empty: return pd.DataFrame()
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

# -------- 사이드바: 카테고리 단일 선택(드롭박스) --------
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
st.sidebar.caption("주변 인프라 선택")
# 공공시설(워킹스페이스)
cb_infra_cafe    = st.sidebar.checkbox("☕ 카페", value=False)
cb_infra_lib     = st.sidebar.checkbox("🏛️ 도서관", value=False라 풍부 지역")
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
    TRANSPORT_FILE = find_optional_file(["transport_access.csv","교통접근성.csv"])
    transport_df = load_transport(TRANSPORT_FILE)
    ktx_df = load_ktx_counts(find_optional_file([
        "한국철도공사_KTX 노선별 역정보_20240411.csv","KTX_노선별_역정보.csv","ktx_stations.csv"
    ]))
    if not transport_df.empty:
        metrics_map = metrics_map.merge(transport_df, on="지역_norm", how="left")
    if not ktx_df.empty:
        metrics_map = metrics_map.merge(ktx_df, on="지역_norm", how="left")
    parts=[]
    if "airport_cnt" in metrics_map:      parts.append(metrics_map["airport_cnt"].rank(pct=True))
    if "ktx_cnt" in metrics_map:          parts.append(metrics_map["ktx_cnt"].rank(pct=True))
    if "bus_term_cnt" in metrics_map:     parts.append(metrics_map["bus_term_cnt"].rank(pct=True))
    if "min_dist_airport" in metrics_map: parts.append(1 - metrics_map["min_dist_airport"].rank(pct=True))
    metrics_map["access_score"] = pd.concat(parts,axis=1).mean(axis=1).clip(0,1) if parts else np.nan

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
    elif selected_category == "💰 합리적인인 비용" and "cost_index" in g:
        rng=(g["cost_index"].max()-g["cost_index"].min())+1e-9
        bonus += CAT_BONUS * (1 - 용" and "cost_index" in g:
        mask = (g["cost_index"] <= g["cost_index"].quantile(0.30))
    elif selected_category == "🚀 빠른 인터넷" and "internet_mbps" in g:
        mask = (g["internet_mbps"] >= g["internet_mbps"].quantile(0.70))
    else:
        mask = pd.Series(True, index=g.index)

    g = g.loc[mask].copy()
    bonus = _compute_bonus_columns(g, selected_category)
    g["NSI"] = (g["NSI_base"] + bonus).clip(0,1)
    return g

metrics_all = apply_category_rules_all(metrics_map)     # 전체용(툴팁)
metrics_after_rules = apply_category_rules(metrics_map) # 필터링 목록/색상용

# ----------------------------- 랭킹 계산 -----------------------------
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
selected_norm = normalize_region_name(st.session_state.selected_region) if st.session_state.selected_region else None

ranked_all = metrics_all.copy()
ranked_all["NSI"]  = ranked_all["NSI"].fillna(ranked_all["NSI_base"]).fillna(0.0).clip(0,1)
ranked_all["rank"] = ranked_all["NSI"].rank(ascending=False, method="min").astype(int)

ranked = metrics_after_rules.copy()
ranked["NSI"]  = ranked["NSI"].fillna(ranked["NSI_base"]).fillna(0.0).clip(0,1)
ranked["rank"] = ranked["NSI"].rank(ascending=False, method="min").astype(int)

# 색: 1·2·3위만 강조
COLOR_TOP1, COLOR_TOP2, COLOR_TOP3 = "#e60049", "#ffd43b", "#4dabf7"
COLOR_SEL, COLOR_BASE = "#51cf66", "#cfd4da"
def pick_color(region_norm, selected_region_norm=None):
    if selected_region_norm and region_norm == selected_region_norm:
        return COLOR_SEL
    r_series = ranked.loc[ranked["지역_norm"]==region_norm, "rank"]
    r = int(r_series.min()) if not r_series.empty else 999
    return {1:COLOR_TOP1, 2:COLOR_TOP2, 3:COLOR_TOP3}.get(r, COLOR_BASE)

# =============================== 지도 ===============================
MAP_HEIGHT = 680
with left:
    m = folium.Map(location=[36.5,127.8], zoom_start=7, tiles="cartodbpositron", prefer_canvas=True)

    gj, gj_err = (None, "no_path")
    if KOREA_GEOJSON and os.path.exists(KOREA_GEOJSON):
        gj, gj_err = load_geojson_safe(KOREA_GEOJSON)

    coords_df = pd.DataFrame([{"지역_norm":k,"lat":v[0],"lon":v[1]} for k,v in REGION_COORDS.items()])
    ranked_all = ranked_all.drop(columns=[c for c in ["lat","lon"] if c in ranked_all.columns]).merge(coords_df, on="지역_norm", how="left")
    rank_lookup = ranked_all.set_index("지역_norm")[["rank","NSI"]].to_dict("index")

    if gj is not None:
        for ft in gj.get("features", []):
            props = ft.get("properties", {}) or {}
            region_raw=None
            for k in GEO_PROP_KEYS:
                if k in props and props[k]: region_raw=props[k]; break
            if region_raw is None:
                textish=[str(v) for v in props.values() if isinstance(v,str)]
                region_raw=max(textish, key=len) if textish else ""
            rname=normalize_region_name(region_raw)
            stats=rank_lookup.get(rname)
            props.update({
                "REGION_NAME": rname,
                "RANK_TXT": f"{int(stats['rank'])}위" if stats else "-",
                "NSI_TXT":  f"{float(stats['NSI']):.3f}" if stats else "-"
            })
            ft["properties"]=props

        def style_function(feature):
            rname = feature["properties"].get("REGION_NAME","")
            color = pick_color(rname, selected_norm)
            return {"fillColor":color, "color":color, "weight":1, "fillOpacity":0.70, "opacity":0.9}

        def highlight_function(feature):
            return {"fillOpacity":0.92, "weight":2}

        tooltip_css = (
            "background-color: rgba(28, 45, 28, 0.92); color: #fff; "
            "font-size: 12px; padding: 6px 8px; border-radius: 6px; "
            "white-space: nowrap; border: 0.5px solid rgba(255,255,255,0.15);"
        )

        GeoJson(
            gj,
            name="regions",
            style_function=style_function,
            highlight_function=highlight_function,
            smooth_factor=1.0,
            tooltip=GeoJsonTooltip(
                fields=["REGION_NAME","RANK_TXT","NSI_TXT"],
                aliases=["지역","랭킹","NSI"],
                labels=True, sticky=True, style=tooltip_css
            ),
            popup=GeoJsonPopup(fields=["REGION_NAME"], labels=False),
        ).add_to(m)

        legend_html = f"""
        <div style="
          position: fixed; bottom: 20px; left: 20px; z-index: 9999;
          background: rgba(255,255,255,0.96); padding: 12px 14px; border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-size: 13px; color: #222;">
          <div style="font-weight:700; margin-bottom:8px;">랭킹 표기</div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP1};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span><span>1위</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP2};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span><span>2위</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP3};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span><span>3위</span>
          </div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        for _, r in ranked_all.iterrows():
            if pd.isna(r.get("lat")) or pd.isna(r.get("lon")): continue
            color = pick_color(r["지역_norm"], selected_norm)
            nsi = float(r["NSI"])
            text = f"지역&nbsp;&nbsp;{r['지역_norm']}<br>랭킹&nbsp;{int(r['rank'])}위<br>NSI&nbsp;&nbsp;&nbsp;{nsi:.3f}"
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=6+14*nsi,
                color=color, fill=True, fill_color=color,
                fill_opacity=0.78, opacity=0.95, weight=1
            ).add_child(folium.Tooltip(text, sticky=True, direction="top")).add_to(m)

    def extract_clicked_name(state):
        if not state: return None
        nm = state.get("last_object_clicked_popup")
        if nm: return normalize_region_name(str(nm))
        obj = state.get("last_active_drawing") or {}
        if isinstance(obj, dict):
            nm = obj.get("properties", {}).get("REGION_NAME")
            if nm: return normalize_region_name(str(nm))
        obj = state.get("last_object_clicked") or {}
        if isinstance(obj, dict):
            nm = obj.get("popup") or (obj.get("properties", {}) or {}).get("REGION_NAME")
            if nm: return normalize_region_name(str(nm))
        return None

    map_state = st_folium(m, width=None, height=MAP_HEIGHT, key="main_map")
    clicked_name = extract_clicked_name(map_state)
    if clicked_name and clicked_name != st.session_state.get("_last_clicked"):
        st.session_state.selected_region = clicked_name
        st.session_state._last_clicked = clicked_name

# ============================ 우측 패널(커뮤니티 유지) ============================
with right:
    st.subheader("커뮤니티")
    c1, c2 = st.columns(2)
    with c1: buddy_on = st.toggle("🧑‍🤝‍🧑 버디 선택", value=False)
    with c2: tourist_on = st.toggle("🧳 관광객 선택", value=False)
    st.caption(f"- 버디: **{'참여' if buddy_on else '미참여'}**  |  관광객: **{'참여' if tourist_on else '미참여'}**")

    st.markdown("### 지역 하이라이트")
    def region_reasons(row, q):
        msgs=[]
        if row.get("방문자_점유율_norm",0) >= q["vis_hi"]: msgs.append("방문 수요가 높아요")
        if "access_score" in row and pd.notna(row["access_score"]) and row["access_score"] >= q["acc_hi"]:
            msgs.append("교통 접근성이 좋아요")
        if "cowork_norm" in row and pd.notna(row["cowork_norm"]) and row["cowork_norm"] >= q["cwk_hi"]:
            msgs.append("공공시설(워킹스페이스)이 발달했어요")
        if not msgs:
            best=[]
            for k, lab in [("방문자_점유율_norm","방문 수요"),
                           ("access_score","교통 접근성"),
                           ("cowork_norm","공공시설(워킹스페이스)")]:
                if k in row: best.append((row[k] if pd.notna(row[k]) else -1, lab))
            best=sorted(best, key=lambda x:x[0], reverse=True)[:2]
            msgs=[f"{lab} 상대적으로 우수" for _,lab in best]
        return " · ".join(msgs)

    q = {
        "vis_hi": ranked_all["방문자_점유율_norm"].quantile(0.70),
        "acc_hi": ranked_all["access_score"].quantile(0.70) if "access_score" in ranked_all and ranked_all["access_score"].notna().any() else 1.0,
        "cwk_hi": ranked_all["cowork_norm"].quantile(0.70) if "cowork_norm" in ranked_all and ranked_all["cowork_norm"].notna().any() else 1.0,
    }

    if st.session_state.selected_region:
        sel = ranked_all.loc[ranked_all["지역_norm"]==normalize_region_name(st.session_state.selected_region)]
        if not sel.empty:
            r=sel.iloc[0]
            st.write(f"**{r['지역_norm']}** — {int(r['rank'])}위 · NSI {float(r['NSI']):.3f}")
            st.caption("· " + region_reasons(r, q))
    else:
        for _, r in ranked_all.sort_values("NSI", ascending=False).head(5).iterrows():
            st.write(f"**{r['지역_norm']}** — {int(r['rank'])}위 · NSI {float(r['NSI']):.3f}")
            st.caption("· " + region_reasons(r, q))

    # QnA/게시판(간단)
    st.markdown("### QnA · 게시판")
    STORE_PATH = os.path.join(DATA_DIR, "community_qna.json")
    def load_store():
        try:
            with open(STORE_PATH,"r",encoding="utf-8") as f: return json.load(f)
        except Exception: return {"posts":[]}
    def save_store(data):
        try:
            with open(STORE_PATH,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        except Exception: pass
    if "qna_store" not in st.session_state: st.session_state.qna_store = load_store()
    store = st.session_state.qna_store

    tabs = st.tabs(["질문 올리기(QnA)", "글쓰기(게시판)", "피드 보기"])
    with tabs[0]:
        with st.form("form_qna"):
            title = st.text_input("제목", value="")
            content = st.text_area("내용", height=120, value="")
            region_tag = st.text_input("관련 지역(선택, 예: 제주·강원)", value=st.session_state.selected_region or "")
            submit = st.form_submit_button("질문 등록")
        if submit and title.strip():
            store["posts"].append({
                "id": str(uuid.uuid4()), "type":"qna",
                "title": title.strip(), "content": content.strip(),
                "region": normalize_region_name(region_tag) if region_tag else "",
                "author":"익명", "created": int(time.time()), "answers":[]
            }); 
            save_store(store); st.success("질문이 등록되었습니다.")
    with tabs[1]:
        with st.form("form_board"):
            title2 = st.text_input("제목 ", value="")
            content2 = st.text_area("본문", height=140, value="")
            region_tag2 = st.text_input("지역 태그(선택)", value=st.session_state.selected_region or "")
            submit2 = st.form_submit_button("글 등록")
        if submit2 and title2.strip():
            store["posts"].append({
                "id": str(uuid.uuid4()), "type":"board",
                "title": title2.strip(), "content": content2.strip(),
                "region": normalize_region_name(region_tag2) if region_tag2 else "",
                "author":"익명", "created": int(time.time()), "comments":[]
            })
            save_store(store); st.success("글이 등록되었습니다.")
    with tabs[2]:
        c1, c2 = st.columns([1,1])
        with c1: feed_type = st.multiselect("유형", ["qna","board"], default=["qna","board"])
        with c2: feed_region = st.text_input("지역 필터(부분일치)", value="")
        posts=[p for p in store["posts"] if p["type"] in feed_type]
        if feed_region.strip():
            key=normalize_region_name(feed_region)
            posts=[p for p in posts if key in normalize_region_name(p.get("region",""))]
        posts=sorted(posts, key=lambda p:p.get("created",0), reverse=True)
        for p in posts:
            with st.expander(f"[{'QnA' if p['type']=='qna' else '게시글'}] {p['title']}  ·  {p.get('region','') or '전체'}"):
                st.write(p["content"] or "(내용 없음)")
                if p["type"]=="qna":
                    for a in p.get("answers", []):
                        st.markdown(f"- **답변**: {a['content']}  — _{a.get('author','익명')}_")
                    with st.form(f"ans_{p['id']}"):
                        ans = st.text_input("답변 달기", value="")
                        if st.form_submit_button("등록") and ans.strip():
                            p.setdefault("answers",[]).append({"content":ans.strip(),"author":"익명","created":int(time.time())})
                            save_store(store); st.success("답변이 등록되었습니다.")
                else:
                    for cmt in p.get("comments", []):
                        st.markdown(f"- **댓글**: {cmt['content']}  — _{cmt.get('author','익명')}_")
                    with st.form(f"cmt_{p['id']}"):
                        cmt = st.text_input("댓글 달기", value="")
                        if st.form_submit_button("등록") and cmt.strip():
                            p.setdefault("comments",[]).append({"content":cmt.strip(),"author":"익명","created":int(time.time())})
                            save_store(store); st.success("댓글이 등록되었습니다.")

# ============================ 랭킹/다운로드 ============================
st.subheader("추천 랭킹")
cols_to_show = ["광역지자체명","NSI","NSI_base","방문자수_합계","방문자_점유율"]
if "access_score" in metrics_map.columns and metrics_map["access_score"].notna().any():
    cols_to_show += ["access_score","ktx_cnt"]
if "coworking_sites" in metrics_map.columns:
    cols_to_show += ["coworking_sites","cowork_per10k"]
if not infra_df.empty:
    cols_to_show += [
        "infra__cafe_count_per10k","infra__library_museum_count_per10k",
        "infra__convenience_count_per10k","infra__laundry_count_per10k",
        "infra__hospital_count_per10k","infra__pharmacy_count_per10k",
        "infra__pc_cafe_count_per10k","infra__karaoke_count_per10k",
        "infra__fitness_count_per10k","infra__yoga_pilates_count_per10k"
    ]
rec = metrics_after_rules.sort_values("NSI", ascending=False)[[c for c in cols_to_show if c in metrics_after_rules.columns]]
out = rec.copy()
if "방문자수_합계" in out.columns: out["방문자수_합계"] = out["방문자수_합계"].fillna(0).astype(int)
for c in out.columns:
    if c not in ["광역지자체명"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
st.dataframe(rec.reset_index(drop=True), use_container_width=True)
st.download_button("⬇️ 랭킹 CSV 저장", out.to_csv(index=False).encode("utf-8-sig"),
                   file_name="ranking_full.csv", mime="text/csv")

# ============================ 키워드 · 카테고리 탐색(그래프) ============================
st.markdown("## 키워드 · 카테고리 탐색")
def load_search_counts(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(), (None, None, None)
    df=None
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try:
            df=pd.read_csv(path, encoding=enc, low_memory=False); break
        except Exception:
            df=None
    if df is None or df.empty: return pd.DataFrame(), (None, None, None)
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

def render_search_chart(df, cols, title_key, default_regions=None, key_prefix="cat"):
    rcol, gcol, vcol = cols
    if df.empty or not (rcol and gcol and vcol):
        st.info("검색건수 데이터를 불러오지 못했습니다. (파일 또는 컬럼 확인)")
        return
    regions = sorted(df[rcol].dropna().astype(str).map(normalize_region_name).unique().tolist())
    if not regions:
        st.info("지역 목록이 비어 있습니다.")
        return
    with st.container():
        c1, c2 = st.columns([2,1])
        with c1:
            pick_regions = st.multiselect("지역 선택", options=regions,
                                          default=default_regions or regions, key=f"{key_prefix}_regions")
        with c2:
            topn = st.slider("상위 N", min_value=5, max_value=30, value=12, step=1, key=f"{key_prefix}_topn")
        temp = df.copy()
        temp["_지역_"] = temp[rcol].astype(str).map(normalize_region_name)
        temp = temp[temp["_지역_"].isin(pick_regions)]
        grp = (temp.groupby(gcol, as_index=False)[vcol].sum()
                    .sort_values(vcol, ascending=False).head(topn))
        st.bar_chart(grp.set_index(gcol)[vcol])

search_cat_df, search_cat_cols   = load_search_counts(file_search_cat)
search_type_df, search_type_cols = load_search_counts(file_search_type)

tabs_kc = st.tabs(["업종/카테고리 검색건수", "유형/키워드 검색건수"])
with tabs_kc[0]:
    render_search_chart(search_cat_df, search_cat_cols, "카테고리", key_prefix="scat")
with tabs_kc[1]:
    render_search_chart(search_type_df, search_type_cols, "키워드", key_prefix="stype")

# ----------------------------- 출처 -----------------------------
st.markdown("""
---
**데이터 출처**  
- 한국관광데이터랩: 지역별 방문자수, 지역별 관광지출액, 지역별 검색건수, 인기관광지 현황  
- 소상공인시장진흥공단: 상가(상권) 정보  
- 한국철도공사: KTX 노선별 역정보  
- 한국문화정보원: 전국공유오피스시설데이터
""")
