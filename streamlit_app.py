# -*- coding: utf-8 -*-
# streamlit run streamlit_app.py
import os, json, re, time, uuid
import streamlit as st
import pandas as pd
import numpy as np

import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
from streamlit_folium import st_folium

st.set_page_config(page_title="디지털 노마드 지역 추천 대시보드", layout="wide")

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CANDIDATE_BASES = [DATA_DIR, APP_DIR, "/mnt/data"]

# ------------------ 세션 상태 ------------------
if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = None
if "_last_clicked" not in st.session_state:
    st.session_state["_last_clicked"] = None

# ------------------ 파일 경로 ------------------
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

def find_optional_file(names):
    for base in CANDIDATE_BASES:
        for n in names:
            p=os.path.join(base,n)
            if os.path.exists(p): return p
    return ""

def resolve_geojson_path():
    env_path = os.environ.get("GEOJSON_PATH", "").strip()
    if env_path and os.path.exists(env_path): return env_path
    for p in [
        os.path.join(DATA_DIR, "korea_provinces.geojson"),
        os.path.join(DATA_DIR, "KOREA_GEOJSON.geojson"),
        os.path.join(APP_DIR,  "korea_provinces.geojson"),
        os.path.join(APP_DIR,  "KOREA_GEOJSON.geojson"),
    ]:
        if os.path.exists(p): return p
    return ""

KOREA_GEOJSON = resolve_geojson_path()
GEO_PROP_KEYS = ["name", "CTPRVN_NM", "ADM1_KOR_NM", "sido_nm", "SIG_KOR_NM", "NAME_1"]

# ------------------ 인프라 소스 탐색 ------------------
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
            if csvs: return {"mode": "dir", "paths": csvs}
    for base in CANDIDATE_BASES:
        z = os.path.join(base, INFRA_ZIP_NAME)
        if os.path.exists(z):
            return {"mode": "zip", "paths": [z]}
    singles = []
    for base in CANDIDATE_BASES:
        if not os.path.isdir(base): continue
        for f in os.listdir(base):
            if f.lower().endswith(".csv") and ("상가" in f or "소상공인" in f):
                singles.append(os.path.join(base, f))
    if singles:
        return {"mode": "single", "paths": singles}
    return {"mode": "none", "paths": []}

# ------------------ 지역명 정규화 ------------------
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
    if not s: return ""
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

# ------------------ 유틸 로더 ------------------
@st.cache_data(show_spinner=False)
def load_geojson_safe(path: str):
    if not path or not os.path.exists(path): return None, "missing_path"
    try:
        if os.path.getsize(path) == 0: return None, "empty_file"
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

@st.cache_data(show_spinner=False)
def read_csv_forgiving(path, usecols=None, dtype=None):
    for enc in ["utf-8","cp949","euc-kr","utf-8-sig","latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, dtype=dtype, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)

# ------------------ 방문자/검색 로더 ------------------
VIS_REGION_KEYS = ["광역지자체명","광역시도","시도","시도명","region","sido","province"]
VIS_COUNT_KEYS  = ["기초지자체 방문자 수","방문자수","방문자 수","합계","total","count"]

@st.cache_data(show_spinner=False)
def read_visitors_flexible(path):
    if not os.path.exists(path): return pd.DataFrame(), (None, None)
    df = read_csv_forgiving(path)
    if df is None or df.empty: return pd.DataFrame(), (None, None)
    cols = {c.lower(): c for c in df.columns}
    def pick(cands):
        for k in cands:
            if k in df.columns: return k
            if k.lower() in cols: return cols[k.lower()]
        return None
    region_col = pick(VIS_REGION_KEYS)
    count_col  = pick(VIS_COUNT_KEYS)
    if not region_col or not count_col:
        string_cols = [c for c in df.columns if df[c].dtype==object]
        num_cols    = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        region_col = region_col or (string_cols[0] if string_cols else None)
        count_col  = count_col  or (num_cols[0] if num_cols else None)
        if not region_col or not count_col:
            return pd.DataFrame(), (None, None)
    df = df[[region_col, count_col]].copy()
    df[region_col] = df[region_col].astype("string")
    df[count_col]  = pd.to_numeric(df[count_col], errors="coerce")
    return df, (region_col, count_col)

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
    rcol=gcol=vcol=None
    for k in ["지역","시도","시도명","광역지자체","sido","region","province"]:
        if k in cols: rcol=cols[k]; break
    for k in ["중분류","대분류","업종","카테고리","유형","키워드","검색어","항목"]:
        if k in cols: gcol=cols[k]; break
    for k in ["검색건수","검색 건수","검색수","검색량","count","건수","value","합계","총건수"]:
        if k in cols: vcol=cols[k]; break
    if not (rcol and gcol and vcol):
        string_cols = [c for c in df.columns if df[c].dtype==object]
        num_cols    = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if len(string_cols)>=2 and len(num_cols)>=1:
            rcol = rcol or string_cols[0]
            gcol = gcol or string_cols[1]
            vcol = vcol or num_cols[0]
        else:
            return pd.DataFrame(), (None, None, None)
    df[vcol]=pd.to_numeric(df[vcol], errors="coerce").fillna(0)
    return df, (rcol,gcol,vcol)

# ======================== 데이터 로딩/전처리 ========================
vis_df_raw, vis_cols = read_visitors_flexible(file_visitors)
if vis_df_raw.empty:
    st.error("방문자 데이터 파일을 불러오지 못했습니다.")
    st.stop()
_region_col, _count_col = vis_cols
vis = (vis_df_raw.groupby(_region_col, as_index=False)[_count_col].sum()
       .rename(columns={_region_col:"광역지자체명", _count_col:"방문자수_합계"}))
vis_region = vis.copy()
total_visitors = max(vis_region["방문자수_합계"].sum(), 1)
vis_region["방문자_점유율"] = vis_region["방문자수_합계"] / total_visitors
vis_region["지역_norm"] = vis_region["광역지자체명"].map(normalize_region_name)

metrics_map = vis_region.copy()
coords_df = pd.DataFrame([{"지역_norm":k,"lat":v[0],"lon":v[1]} for k,v in REGION_COORDS.items()])
metrics_map = metrics_map.merge(coords_df, on="지역_norm", how="left")

metrics_map["방문자_점유율_norm"] = minmax(metrics_map["방문자_점유율"].fillna(0))

# ── 검색데이터 기반 숙박 비중(%) ──
s_cat_df, s_cat_cols = load_search_counts(file_search_cat)
if not s_cat_df.empty and all(s_cat_cols):
    rcol, gcol, vcol = s_cat_cols
    tmp = s_cat_df.copy()
    tmp["_지역_"] = tmp[rcol].astype(str).map(normalize_region_name)
    tmp["_카테고리_"] = tmp[gcol].astype(str)
    tmp[vcol] = pd.to_numeric(tmp[vcol], errors="coerce").fillna(0)

    total = (tmp.groupby("_지역_", as_index=False)[vcol]
                .sum().rename(columns={vcol: "_total_"}))

    lodg_mask = tmp["_카테고리_"].str.contains(
        r"숙박|호텔|모텔|게스트하우스|게하|호스텔|리조트|펜션|민박|B&B|bnb",
        case=False, na=False
    )
    lodg = (tmp[lodg_mask].groupby("_지역_", as_index=False)[vcol]
                .sum().rename(columns={vcol: "_lodging_"}))

    lodg_share = total.merge(lodg, on="_지역_", how="left")
    lodg_share["_lodging_"] = lodg_share["_lodging_"].fillna(0)
    lodg_share["숙박_지출비중(%)"] = (
        (lodg_share["_lodging_"] / lodg_share["_total_"]).replace([np.inf, np.nan], 0) * 100
    )

    metrics_map = metrics_map.drop(
        columns=[c for c in ["숙박_지출비중(%)","숙박_비중_norm","NSI_base"] if c in metrics_map.columns]
    ).merge(
        lodg_share[["_지역_", "숙박_지출비중(%)"]].rename(columns={"_지역_":"지역_norm"}),
        on="지역_norm", how="left"
    )
else:
    metrics_map["숙박_지출비중(%)"] = np.nan

metrics_map["숙박_비중_norm"] = minmax(metrics_map["숙박_지출비중(%)"].fillna(0))
metrics_map["NSI_base"] = 0.60*metrics_map["방문자_점유율_norm"] + 0.40*metrics_map["숙박_비중_norm"]

# ==================== 생활 인프라(항상 시도) ====================
REQUIRED_ALIASES = {
    "시도명": ["시도명","시도","광역시도","광역지자체","행정구역명","시도명칭"],
    "상권업종중분류명": ["상권업종중분류명","중분류명","상권업종중분류코드명"],
    "상권업종소분류명": ["상권업종소분류명","소분류명","상권업종소분류코드명"],
    "표준산업분류명": ["표준산업분류명","표준산업분류코드명","산업분류명"]
}
def _map_columns_loose(df):
    colmap = {}
    low = {c.lower(): c for c in df.columns}
    for need, aliases in REQUIRED_ALIASES.items():
        hit = None
        for cand in aliases:
            if cand in df.columns:
                hit = cand; break
            if cand.lower() in low:
                hit = low[cand.lower()]; break
        if not hit:
            for c in df.columns:
                if any(x.lower() in c.lower() for x in aliases):
                    hit = c; break
        if not hit:
            return None
        colmap[need] = hit
    return colmap

@st.cache_data(show_spinner=True)
def build_infra_from_sources(sources):
    import io, zipfile
    dfs = []
    def try_read_csv(path):
        df=None
        for enc in ["cp949","utf-8","euc-kr","latin1","utf-8-sig"]:
            try:
                df=pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except Exception:
                df=None
        return df
    def std_df(df):
        colmap=_map_columns_loose(df)
        if not colmap: return None
        df=df.rename(columns=colmap)[list(colmap.keys())].copy()
        for c in df.columns: df[c]=df[c].astype(str).str.strip()
        return df

    if sources["mode"] == "dir":
        for path in sources["paths"]:
            df = try_read_csv(path)
            df = std_df(df) if df is not None else None
            if df is not None: dfs.append(df)
    elif sources["mode"] == "zip":
        zpath = sources["paths"][0]
        with zipfile.ZipFile(zpath,"r") as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"): continue
                raw=z.read(name); df=None
                for enc in ["cp949","utf-8","euc-kr","latin1","utf-8-sig"]:
                    try:
                        df=pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False); break
                    except Exception:
                        df=None
                df = std_df(df) if df is not None else None
                if df is not None: dfs.append(df)
    elif sources["mode"] == "single":
        for path in sources["paths"]:
            df = try_read_csv(path)
            df = std_df(df) if df is not None else None
            if df is not None: dfs.append(df)

    if not dfs: return pd.DataFrame()

    df=pd.concat(dfs, ignore_index=True)
    mid=df["상권업종중분류명"].astype(str)
    sub=df["상권업종소분류명"].astype(str)
    std=df["표준산업분류명"].astype(str)

    m_cafe  = (sub.str.contains("카페")) | (std.str.contains("커피 전문점"))
    m_conv  = (sub.str.contains("편의점")) | (std.str.contains("체인화 편의점"))
    m_hotel = sub.str.contains("호텔/리조트"); m_motel=sub.str.contains("여관/모텔"); m_accom_mid=mid.str.contains("숙박")
    m_pc    = sub.str_contains("PC방", regex=False)
    m_lndry = sub.str.contains("세탁소|빨래방")
    m_pharm = sub.str.contains("약국")
    m_clinic= mid.str.contains("의원")
    m_hosp  = mid.str.contains("병원") | m_clinic | sub.str.contains("치과의원|한의원|내과|외과|피부|비뇨")
    m_lib   = mid.str.contains("도서관·사적지|도서관")

    df["sido_norm"]=df["시도명"].map(normalize_region_name)
    agg=df.groupby("sido_norm").agg(
        total_places=("시도명","size"),
        cafe_count=("시도명", lambda s:int(m_cafe.loc[s.index].sum())),
        convenience_count=("시도명", lambda s:int(m_conv.loc[s.index].sum())),
        accommodation_count=("시도명", lambda s:int((m_hotel|m_motel|m_accom_mid).loc[s.index].sum())),
        hospital_count=("시도명", lambda s:int(m_hosp.loc[s.index].sum())),
        pharmacy_count=("시도명", lambda s:int(m_pharm.loc[s.index].sum())),
        pc_cafe_count=("시도명", lambda s:int(m_pc.loc[s.index].sum())),
        laundry_count=("시도명", lambda s:int(m_lndry.loc[s.index].sum())),
        library_museum_count=("시도명", lambda s:int(m_lib.loc[s.index].sum())),
    ).reset_index()

    def per_10k(n,total): total=total.replace(0,np.nan); return (n/total)*10000
    for col in ["cafe_count","convenience_count","accommodation_count","hospital_count",
                "pharmacy_count","pc_cafe_count","laundry_count","library_museum_count"]:
        agg[col+"_per10k"] = per_10k(agg[col].astype(float), agg["total_places"].astype(float)).round(3)
        v=agg[col].astype(float); rng=v.max()-v.min()
        agg[col+"_norm"] = ((v-v.min())/rng).fillna(0).round(4) if rng>0 else v*0
    return agg

infra_sources = resolve_infra_sources()
infra_df = build_infra_from_sources(infra_sources)
if not infra_df.empty:
    metrics_map = metrics_map.merge(
        infra_df.add_prefix("infra__").rename(columns={"infra__sido_norm":"지역_norm"}),
        on="지역_norm", how="left"
    )

with st.expander("🔧 인프라 데이터 진단"):
    st.write("infra_sources →", infra_sources)
    if infra_df.empty:
        st.error("infra_df가 비어 있습니다. (폴더/ZIP 경로 또는 컬럼명 불일치 가능)")
    else:
        st.success(f"infra_df 로드 OK · 행 {len(infra_df)}")
        st.write(infra_df.head(3))
        miss_cols = [c for c in [
            "infra__cafe_count_per10k","infra__convenience_count_per10k","infra__accommodation_count_per10k",
            "infra__hospital_count_per10k","infra__pharmacy_count_per10k","infra__library_museum_count_per10k"
        ] if c not in metrics_map.columns]
        if miss_cols:
            st.warning(f"merge 후 metrics_map에 없는 인프라 컬럼: {miss_cols}")
        null_rate = metrics_map.filter(like="infra__").isna().mean().sort_values(ascending=False).head(10)
        st.write("상위 결측률", null_rate)
        st.write("인프라 쪽 지역_norm 샘플:", infra_df["sido_norm"].drop_duplicates().head(10).tolist())
        st.write("메트릭 쪽 지역_norm 샘플:", metrics_map["지역_norm"].drop_duplicates().head(10).tolist())

# ==================== 교통 접근성: KTX만 사용 ====================
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

ktx_df = load_ktx_counts(find_optional_file([
    "한국철도공사_KTX 노선별 역정보_20240411.csv","KTX_노선별_역정보.csv","ktx_stations.csv"
]))
if not ktx_df.empty:
    metrics_map = metrics_map.merge(ktx_df, on="지역_norm", how="left")
    metrics_map["access_score"] = metrics_map["ktx_cnt"].rank(pct=True).clip(0,1)
else:
    metrics_map["access_score"] = np.nan

# ============================ UI 고정 요소 ============================
st.title("디지털 노마드 지역 추천 대시보드")
left, right = st.columns([2, 1])
with left:
    st.subheader("지도에서 지역을 선택하세요")

st.sidebar.header("추천 카테고리")
CATEGORY_OPTIONS = [
    "🔥 현재 인기 지역","🛏️ 숙박 다양 지역","🚉 교통 좋은 지역",
    "💼 코워킹 인프라 풍부 지역","💰 저렴한 비용","🚀 빠른 인터넷",
]
selected_category = st.sidebar.selectbox("하나만 선택", CATEGORY_OPTIONS, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("주변 인프라(대분류) 보너스 반영")
medical_cb     = st.sidebar.checkbox("🧑‍⚕️ 의료시설", value=False)
convenience_cb = st.sidebar.checkbox("🛒 편의시설", value=False)
workspace_cb   = st.sidebar.checkbox("💼 워킹 스페이스", value=False)
leisure_cb     = st.sidebar.checkbox("🎽 여가·운동", value=False)
lodging_cb     = st.sidebar.checkbox("🏨 숙박", value=False)

# ----------------------------- 점수 산출 -----------------------------
def _compute_bonus_columns(g, selected_category):
    CAT_BONUS   = 0.15
    INFRA_BONUS = 0.10
    q_vis_hi = g["방문자_점유율_norm"].dropna().quantile(0.70) if g["방문자_점유율_norm"].notna().any() else 1.0
    q_lod_hi = g["숙박_비중_norm"].dropna().quantile(0.70)     if g["숙박_비중_norm"].notna().any()     else 1.0
    q_acc_hi = g["access_score"].dropna().quantile(0.70)        if "access_score" in g and g["access_score"].notna().any() else 1.0
    q_cwk_hi = g["cowork_norm"].dropna().quantile(0.70)         if "cowork_norm" in g and g["cowork_norm"].notna().any()   else 1.0
    bonus = np.zeros(len(g), dtype=float)
    def add_above(series, q):
        s=pd.to_numeric(series, errors="coerce").fillna(0)
        return (s - q).clip(lower=0)
    if selected_category == "🔥 현재 인기 지역":
        bonus += CAT_BONUS * add_above(g["방문자_점유율_norm"], q_vis_hi)
    elif selected_category == "🛏️ 숙박 다양 지역":
        bonus += CAT_BONUS * add_above(g["숙박_비중_norm"], q_lod_hi)
    elif selected_category == "🚉 교통 좋은 지역" and "access_score" in g:
        bonus += CAT_BONUS * add_above(g["access_score"], q_acc_hi)
    elif selected_category == "💼 코워킹 인프라 풍부 지역" and "cowork_norm" in g:
        bonus += CAT_BONUS * add_above(g["cowork_norm"], q_cwk_hi)

    def has(col): return col in g.columns and pd.api.types.is_numeric_dtype(g[col])
    infra_cols = {
        "cafe":"infra__cafe_count_norm", "conv":"infra__convenience_count_norm",
        "accom":"infra__accommodation_count_norm", "hosp":"infra__hospital_count_norm",
        "pharm":"infra__pharmacy_count_norm", "pc":"infra__pc_cafe_count_norm",
        "laundry":"infra__laundry_count_norm", "lib":"infra__library_museum_count_norm",
    }
    if medical_cb:
        if has(infra_cols["hosp"]):   bonus += INFRA_BONUS * g[infra_cols["hosp"]].fillna(0)
        if has(infra_cols["pharm"]):  bonus += INFRA_BONUS * g[infra_cols["pharm"]].fillna(0)
    if convenience_cb:
        if has(infra_cols["conv"]):    bonus += INFRA_BONUS * g[infra_cols["conv"]].fillna(0)
        if has(infra_cols["laundry"]): bonus += INFRA_BONUS * g[infra_cols["laundry"]].fillna(0)
    if workspace_cb:
        if has(infra_cols["cafe"]): bonus += INFRA_BONUS * g[infra_cols["cafe"]].fillna(0)
        if has(infra_cols["lib"]):  bonus += INFRA_BONUS * g[infra_cols["lib"]].fillna(0)
    if leisure_cb and has(infra_cols["pc"]): bonus += INFRA_BONUS * g[infra_cols["pc"]].fillna(0)
    if lodging_cb and has(infra_cols["accom"]): bonus += INFRA_BONUS * g[infra_cols["accom"]].fillna(0)
    return bonus

def apply_category_rules_all(df):
    g = df.copy()
    # 코워킹 정규화: 인프라 표본 있으면 per10k 기반, 없으면 raw count 기반
    if "cowork_per10k" in g and pd.to_numeric(g["cowork_per10k"], errors="coerce").notna().any():
        v = pd.to_numeric(g["cowork_per10k"], errors="coerce")
        rng=(v.max()-v.min()); g["cowork_norm"]=((v-v.min())/rng).fillna(0) if rng>0 else v*0
    elif "coworking_sites" in g:
        v = pd.to_numeric(g["coworking_sites"], errors="coerce")
        g["cowork_norm"]=v.rank(pct=True).fillna(0)
    bonus = _compute_bonus_columns(g, selected_category)
    g["NSI"] = (g["NSI_base"] + bonus).clip(0,1)
    return g

def apply_category_rules(df):
    g = df.copy()
    q_vis_hi = g["방문자_점유율_norm"].dropna().quantile(0.70) if g["방문자_점유율_norm"].notna().any() else 1.0
    q_lod_hi = g["숙박_비중_norm"].dropna().quantile(0.70)     if g["숙박_비중_norm"].notna().any()     else 1.0
    q_acc_hi = g["access_score"].dropna().quantile(0.70)        if "access_score" in g and g["access_score"].notna().any() else 1.0
    q_cwk_hi = g["cowork_norm"].dropna().quantile(0.70)         if "cowork_norm" in g and g["cowork_norm"].notna().any()   else 1.0
    if selected_category == "🔥 현재 인기 지역":
        mask = (g["방문자_점유율_norm"] >= q_vis_hi)
    elif selected_category == "🛏️ 숙박 다양 지역":
        mask = (g["숙박_비중_norm"] >= q_lod_hi)
    elif selected_category == "🚉 교통 좋은 지역" and "access_score" in g:
        mask = (g["access_score"] >= q_acc_hi)
    elif selected_category == "💼 코워킹 인프라 풍부 지역" and "cowork_norm" in g:
        mask = (g["cowork_norm"] >= q_cwk_hi)
    else:
        mask = pd.Series(True, index=g.index)
    g = g.loc[mask].copy()
    # 코워킹 정규화 동일 적용
    if "cowork_norm" not in g.columns:
        if "cowork_per10k" in g:
            v = pd.to_numeric(g["cowork_per10k"], errors="coerce"); rng=(v.max()-v.min())
            g["cowork_norm"]=((v-v.min())/rng).fillna(0) if rng>0 else v*0
        elif "coworking_sites" in g:
            g["cowork_norm"]=pd.to_numeric(g["coworking_sites"], errors="coerce").rank(pct=True).fillna(0)
    bonus = _compute_bonus_columns(g, selected_category)
    g["NSI"] = (g["NSI_base"] + bonus).clip(0,1)
    return g

# 코워킹: 데이터가 있으면 per10k 계산(인프라 total_places 필요)
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

# 코워킹 파일 있으면 반영
cw_file = find_optional_file([
    "KC_CNRS_OFFM_FCLTY_DATA_2023.csv","공유오피스.csv","coworking_sites.csv",
    "중소벤처기업진흥공단_공유오피스_운영현황.csv","한국문화정보원_전국공유오피스시설.csv","전국_공유_오피스_시설_데이터.csv"
])
cow_df = load_coworking(cw_file)
if not cow_df.empty:
    if "infra__total_places" in metrics_map.columns:
        base = metrics_map[["지역_norm","infra__total_places"]]
        cow = cow_df.merge(base, on="지역_norm", how="left")
        cow["cowork_per10k"] = (cow["coworking_sites"]/cow["infra__total_places"].replace(0,np.nan)*10000).round(3)
    else:
        cow = cow_df.copy()
        cow["cowork_per10k"]=np.nan
    metrics_map = metrics_map.merge(cow[["지역_norm","coworking_sites","cowork_per10k"]], on="지역_norm", how="left")

# ----------------------------- 최종 점수 데이터 -----------------------------
metrics_all = apply_category_rules_all(metrics_map)
metrics_after_rules = apply_category_rules(metrics_map)

def normalized(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    mm = minmax(s)
    return (1-mm) if invert else mm

def category_display_score(df, category):
    g = df.copy()
    score = None
    if category == "🔥 현재 인기 지역":
        score = g["방문자_점유율_norm"]
    elif category == "🛏️ 숙박 다양 지역":
        score = g["숙박_비중_norm"]
    elif category == "🚉 교통 좋은 지역":
        score = normalized(g.get("access_score", np.nan))
    elif category == "💼 코워킹 인프라 풍부 지역":
        if "cowork_norm" in g:        score = g["cowork_norm"]
        elif "cowork_per10k" in g:    score = normalized(g["cowork_per10k"])
        elif "coworking_sites" in g:  score = g["coworking_sites"].rank(pct=True)
    if score is None or pd.to_numeric(score, errors="coerce").fillna(0).nunique() <= 1:
        score = g["NSI"]
    return pd.to_numeric(score, errors="coerce").fillna(0).clip(0,1)

ranked_all = metrics_all.copy()
ranked_all["NSI"]  = ranked_all["NSI"].fillna(ranked_all["NSI_base"]).fillna(0.0).clip(0,1)
ranked_view = ranked_all.copy()
ranked_view["display_score"] = category_display_score(ranked_all, selected_category)
ranked_view["rank_view"]     = ranked_view["display_score"].rank(ascending=False, method="min").astype(int)

# =============================== 지도 ===============================
COLOR_TOP1, COLOR_TOP2, COLOR_TOP3 = "#e60049", "#ffd43b", "#4dabf7"
COLOR_SEL, COLOR_BASE = "#51cf66", "#cfd4da"

TOP3 = (
    ranked_view
    .sort_values(["display_score", "지역_norm"], ascending=[False, True])
    .dropna(subset=["display_score"])
    .head(3)["지역_norm"]
    .tolist()
)
TOP3_COLORS = [COLOR_TOP1, COLOR_TOP2, COLOR_TOP3]
TOP3_COLOR_MAP = {name: TOP3_COLORS[i] for i, name in enumerate(TOP3)}

def pick_color(region_norm, selected_region_norm=None):
    if selected_region_norm and region_norm == selected_region_norm:
        return COLOR_SEL
    return TOP3_COLOR_MAP.get(region_norm, COLOR_BASE)

MAP_HEIGHT = 680
with left:
    m = folium.Map(location=[36.5,127.8], zoom_start=7, tiles="cartodbpositron", prefer_canvas=True)
    gj, gj_err = (None, "no_path")
    if KOREA_GEOJSON and os.path.exists(KOREA_GEOJSON):
        gj, gj_err = load_geojson_safe(KOREA_GEOJSON)
    coords_df = pd.DataFrame([{"지역_norm":k,"lat":v[0],"lon":v[1]} for k,v in REGION_COORDS.items()])
    ranked_view = ranked_view.drop(columns=[c for c in ["lat","lon"] if c in ranked_view.columns]).merge(coords_df, on="지역_norm", how="left")
    rank_lookup = ranked_view.set_index("지역_norm")[["rank_view","display_score"]].to_dict("index")
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
                "RANK_TXT": f"{int(stats['rank_view'])}위" if stats else "-",
                "SCORE_TXT": f"{float(stats['display_score']):.3f}" if stats else "-"
            })
            ft["properties"]=props
        def style_function(feature):
            rname = feature["properties"].get("REGION_NAME","")
            color = pick_color(rname, normalize_region_name(st.session_state.get("selected_region")))
            return {"fillColor":color, "color":color, "weight":1, "fillOpacity":0.70, "opacity":0.9}
        def highlight_function(feature):
            return {"fillOpacity":0.92, "weight":2}
        tooltip_css = ("background-color: rgba(28, 45, 28, 0.92); color: #fff; "
                       "font-size: 12px; padding: 6px 8px; border-radius: 6px; "
                       "white-space: nowrap; border: 0.5px solid rgba(255,255,255,0.15);")
        GeoJson(
            gj,
            name="regions",
            style_function=style_function,
            highlight_function=highlight_function,
            smooth_factor=1.0,
            tooltip=GeoJsonTooltip(
                fields=["REGION_NAME","RANK_TXT","SCORE_TXT"],
                aliases=["지역","랭킹","점수"],
                labels=True, sticky=True, style=tooltip_css
            ),
            popup=GeoJsonPopup(fields=["REGION_NAME"], labels=False),
        ).add_to(m)
        legend_html = f"""
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
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
    map_state = st_folium(m, width=None, height=MAP_HEIGHT, key="main_map")
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
    clicked_name = extract_clicked_name(map_state)
    if clicked_name and clicked_name != st.session_state.get("_last_clicked"):
        st.session_state["selected_region"] = clicked_name
        st.session_state["_last_clicked"] = clicked_name

# ============================ 우측 패널 ============================
with right:
    st.subheader("커뮤니티")
    st.markdown("### 지역 하이라이트")
    if st.session_state.get("selected_region"):
        sel_name = normalize_region_name(st.session_state.get("selected_region"))
        sel = ranked_view.loc[ranked_view["지역_norm"]==sel_name]
        if not sel.empty:
            r=sel.iloc[0]
            st.info(f"**선택 지역: {r['지역_norm']}** — {int(r['rank_view'])}위 · 점수 {float(r['display_score']):.3f}")
    for _, r in ranked_view.sort_values("display_score", ascending=False).head(5).iterrows():
        name = r["지역_norm"]
        strong = "**" if st.session_state.get("selected_region") and normalize_region_name(st.session_state.get("selected_region"))==name else ""
        st.write(f"{strong}{name}{strong} — {int(r['rank_view'])}위 · 점수 {float(r['display_score']):.3f}")

    st.markdown("### QnA · 게시판")
    STORE_PATH = os.path.join(DATA_DIR, "community_qna.json")
    def load_store():
        try:
            with open(STORE_PATH,"r",encoding="utf-8") as f: return json.load(f)
        except Exception: return {"posts":[]}
    def save_store(data):
        try:
            with open(STORE_PATH,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        except Exception:
            st.warning("게시글 저장에 실패했습니다. 쓰기 권한을 확인하세요.")
    if "qna_store" not in st.session_state: st.session_state["qna_store"] = load_store()
    store = st.session_state["qna_store"]

    tabs = st.tabs(["질문 올리기(QnA)", "글쓰기(게시판)", "피드 보기"])
    with tabs[0]:
        with st.form("form_qna"):
            title = st.text_input("제목", value="")
            content = st.text_area("내용", height=120, value="")
            region_tag = st.text_input("관련 지역(선택, 예: 제주·강원)", value=st.session_state.get("selected_region","") or "")
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
            region_tag2 = st.text_input("지역 태그(선택)", value=st.session_state.get("selected_region","") or "")
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

# ============================ 🔎 핵심지표 현황 ============================
st.markdown("## 🔎 핵심지표 현황 (전국 평균 & 선택 지역 비교)")
core_metric_labels = {
    "방문자_점유율": "방문자 점유율",
    "숙박_지출비중(%)": "숙박 지출 비중(%)",
    "access_score": "교통 접근성(정규화, KTX만)",
    "infra__cafe_count_per10k": "카페/커피(1만 개당)",
    "infra__convenience_count_per10k": "편의점(1만 개당)",
    "infra__accommodation_count_per10k": "숙박(1만 개당)",
    "infra__hospital_count_per10k": "병원(1만 개당)",
    "infra__pharmacy_count_per10k": "약국(1만 개당)",
    "infra__library_museum_count_per10k": "도서관·사적지(1만 개당)",
    "NSI": "NSI(가중 보너스)",
    "NSI_base": "NSI_base(기본)"
}
def pick_existing(cols, df):
    return [c for c in cols if c in df.columns]

num_cols = pick_existing(list(core_metric_labels.keys()), ranked_all)
if not num_cols:
    st.info("핵심지표를 계산할 수 있는 컬럼이 없어 요약을 표시하지 않습니다.")
else:
    core_df = ranked_all[["지역_norm"] + num_cols].copy()
    desc = core_df[num_cols].describe().T.rename(columns={
        "mean": "평균", "std": "표준편차", "50%": "중앙값",
        "min": "최솟값", "max": "최댓값", "25%": "하위 25%", "75%": "상위 25%"
    })
    desc.index = [core_metric_labels.get(c, c) for c in desc.index]

    st.markdown("### 📍 선택 지역 vs 전국 평균")
    sel = None
    if st.session_state.get("selected_region"):
        key = normalize_region_name(st.session_state["selected_region"])
        tmp = core_df.loc[core_df["지역_norm"] == key]
        if not tmp.empty:
            sel = tmp.iloc[0]

    rows = (len(num_cols) + 1) // 2
    for r in range(rows):
        c1, c2 = st.columns(2)
        for i, col in enumerate(num_cols[r*2 : r*2+2]):
            container = c1 if i == 0 else c2
            with container:
                label = core_metric_labels.get(col, col)
                avg = core_df[col].mean()

                def fmt(v, colname):
                    if pd.isna(v): return "-"
                    if "지출비중" in colname or "점유율" in colname:
                        return f"{float(v)*100:.2f}%" if core_df[colname].max() <= 1.0 else f"{float(v):.2f}%"
                    return f"{float(v):.3f}"

                if sel is not None:
                    val = sel[col]
                    delta = (val - avg)
                    if ("지출비중" in col) or ("점유율" in col):
                        if core_df[col].max() <= 1.0:
                            st.metric(label, f"{val*100:.2f}%", delta=f"{delta*100:+.2f}%p")
                        else:
                            st.metric(label, f"{val:.2f}%", delta=f"{delta:+.2f}%p")
                    else:
                        st.metric(label, fmt(val, col), delta=f"{delta:+.3f}")
                else:
                    st.metric(label + " (전국 평균)", fmt(avg, col))

    st.markdown("### 📊 전국 분포 요약")
    st.dataframe(desc[["평균", "중앙값", "표준편차", "하위 25%", "상위 25%", "최솟값", "최댓값"]].round(4), use_container_width=True)

    with st.expander("분포 보기(히스토그램)"):
        sel_cols = st.multiselect("분포를 볼 지표 선택", options=num_cols,
                                  format_func=lambda c: core_metric_labels.get(c, c),
                                  default=num_cols[:2])
        for col in sel_cols:
            st.caption(f"• {core_metric_labels.get(col, col)}")
            st.bar_chart(core_df[col].sort_values().reset_index(drop=True))

# ============================ 랭킹/다운로드 ============================
st.subheader("추천 랭킹 (Top 5)")
cols_to_show = ["광역지자체명","display_score","NSI","NSI_base",
                "방문자수_합계","방문자_점유율","숙박_지출비중(%)","access_score","ktx_cnt"]
if "coworking_sites" in metrics_map.columns:
    cols_to_show += ["coworking_sites","cowork_per10k"]
if 'infra__cafe_count_per10k' in metrics_after_rules.columns:
    cols_to_show += [
        "infra__cafe_count_per10k","infra__convenience_count_per10k","infra__accommodation_count_per10k",
        "infra__hospital_count_per10k","infra__pharmacy_count_per10k","infra__library_museum_count_per10k"
    ]
rec = ranked_view.sort_values("display_score", ascending=False)[[c for c in cols_to_show if c in ranked_view.columns]]
top5 = rec.head(5)
out = rec.copy()
if "방문자수_합계" in out.columns: out["방문자수_합계"] = out["방문자수_합계"].fillna(0).astype(int)
for c in out.columns:
    if c not in ["광역지자체명"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
st.dataframe(top5.reset_index(drop=True), use_container_width=True)
st.download_button("⬇️ 전체 랭킹 CSV 저장", out.to_csv(index=False).encode("utf-8-sig"),
                   file_name="ranking_full.csv", mime="text/csv")

# ============================ 키워드 · 카테고리 탐색 ============================
st.markdown("## 키워드 · 카테고리 탐색")
search_cat_df, search_cat_cols   = load_search_counts(file_search_cat)
search_type_df, search_type_cols = load_search_counts(file_search_type)

def render_search_chart(df, cols, title_key, default_regions=None, key_prefix="cat"):
    rcol, gcol, vcol = cols
    if df.empty or not (rcol and gcol and vcol):
        st.info("검색건수 데이터를 불러오지 못했습니다. (파일 또는 컬럼 확인)")
        return
    regions = sorted(df[rcol].dropna().astype(str).map(normalize_region_name).unique().tolist())
    if not regions:
        st.info("지역 목록이 비어 있습니다."); return
    with st.container():
        c1, c2 = st.columns([2,1])
        default_sel = default_regions or ([st.session_state.get("selected_region")] if st.session_state.get("selected_region") else regions)
        with c1:
            pick_regions = st.multiselect("지역 선택", options=regions,
                                          default=default_sel, key=f"{key_prefix}_regions")
        with c2:
            topn = st.slider("상위 N", min_value=5, max_value=30, value=5, step=1, key=f"{key_prefix}_topn")
        temp = df.copy()
        temp["_지역_"] = temp[rcol].astype(str).map(normalize_region_name)
        temp[vcol] = pd.to_numeric(temp[vcol], errors="coerce").fillna(0)
        temp = temp[temp["_지역_"].isin([normalize_region_name(r) for r in pick_regions])]
        grp = (temp.groupby(gcol, as_index=False)[vcol].sum()
                    .sort_values(vcol, ascending=False).head(topn))
        st.bar_chart(grp.set_index(gcol)[vcol])

tabs_kc = st.tabs(["업종/카테고리 검색건수", "유형/키워드 검색건수"])
with tabs_kc[0]:
    render_search_chart(search_cat_df, search_cat_cols, "카테고리", key_prefix="scat")
with tabs_kc[1]:
    render_search_chart(search_type_df, search_type_cols, "키워드", key_prefix="stype")

st.markdown("### 선택 지역 카테고리/키워드 Top 5")
def topn_by_region(df, cols, region_norm, topn=5):
    rcol, gcol, vcol = cols
    if df.empty or not (rcol and gcol and vcol):
        return pd.DataFrame(), None, None
    temp = df.copy()
    temp["_지역_"] = temp[rcol].astype(str).map(normalize_region_name)
    temp[vcol] = pd.to_numeric(temp[vcol], errors="coerce")
    temp = temp[temp["_지역_"]==region_norm]
    if temp.empty: return pd.DataFrame(), None, None
    grp = (temp.groupby(gcol, as_index=False)[vcol].sum()
                .sort_values(vcol, ascending=False).head(topn))
    return grp, gcol, vcol

if st.session_state.get("selected_region"):
    _sel = normalize_region_name(st.session_state.get("selected_region"))
    c1, c2 = st.columns(2)
    with c1:
        grp, gcol, vcol = topn_by_region(search_cat_df, search_cat_cols, _sel, topn=5)
        if grp.empty:
            st.info("선택 지역의 업종/카테고리 검색 데이터가 없습니다.")
        else:
            st.subheader("업종/카테고리 Top 5")
            st.bar_chart(grp.set_index(gcol)[vcol])
    with c2:
        grp2, gcol2, vcol2 = topn_by_region(search_type_df, search_type_cols, _sel, topn=5)
        if grp2.empty:
            st.info("선택 지역의 유형/키워드 검색 데이터가 없습니다.")
        else:
            st.subheader("유형/키워드 Top 5")
            st.bar_chart(grp2.set_index(gcol2)[vcol2])
else:
    st.caption("지역을 선택하면 해당 지역의 상위 카테고리/키워드를 5개 보여드립니다.")
