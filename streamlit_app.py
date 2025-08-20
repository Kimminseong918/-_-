# -*- coding: utf-8 -*-
import os
import json
import re
import time
import uuid
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
CANDIDATE_BASES = [DATA_DIR, APP_DIR, "/mnt/data"]   # 집 폴더 미사용

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

file_visitors, file_spend_cat, file_spend_type = build_paths()

def resolve_geojson_path():
    env_path = os.environ.get("GEOJSON_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        os.path.join(DATA_DIR, "korea_provinces.geojson"),
        os.path.join(DATA_DIR, "KOREA_GEOJSON.geojson"),
        os.path.join(APP_DIR,  "korea_provinces.geojson"),
        os.path.join(APP_DIR,  "KOREA_GEOJSON.geojson"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

KOREA_GEOJSON = resolve_geojson_path()
GEO_PROP_KEYS = ["name", "CTPRVN_NM", "ADM1_KOR_NM", "sido_nm", "SIG_KOR_NM", "NAME_1"]

# ---- 인프라 폴더/ZIP 자동 탐색(집 폴더 X) ----
INFRA_DIR_NAME  = "소상공인시장진흥공단_상가(상권)정보_20250630"
INFRA_ZIP_NAME  = "소상공인시장진흥공단_상가(상권)정보_20250630.zip"

def resolve_infra_sources():
    # 폴더 우선 → ZIP
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
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp949", "euc-kr", "latin-1")
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                txt = f.read().strip()
            if not txt:
                return None, f"empty_text({enc})"
            if txt.lstrip().startswith(("var ", "const ", "let ")):
                m = re.search(r"\{.*\}\s*;?\s*$", txt, flags=re.S)
                if m:
                    txt = m.group(0)
            gj = json.loads(txt)
            if not isinstance(gj, dict): return None, f"not_dict({enc})"
            if gj.get("type") == "Topology": return None, "topojson_not_supported"
            if gj.get("type") != "FeatureCollection" or not isinstance(gj.get("features"), list):
                return None, f"not_featurecollection({enc})"
            return gj, None
        except Exception as e:
            last_err = f"{enc}: {e}"
            continue
    return None, last_err or "unknown_error"

# ---------------------------- 표준/약식 지역명 ----------------------------
# 표시용 2글자 약식명으로 통일 (툴팁 잘림 방지)
TWOCHAR_MAP = {
    "서울":"서울","부산":"부산","대구":"대구","인천":"인천","광주":"광주","대전":"대전","울산":"울산","세종":"세종",
    "경기":"경기","강원":"강원","충북":"충북","충남":"충남","전북":"전북","전남":"전남","경북":"경북","경남":"경남","제주":"제주",
}
def to_twochar(s: str) -> str:
    s = str(s)
    if s.startswith("전라남"): return "전남"
    if s.startswith("전라북"): return "전북"
    if s.startswith("경상남"): return "경남"
    if s.startswith("경상북"): return "경북"
    if s.startswith("충청남"): return "충남"
    if s.startswith("충청북"): return "충북"
    # 그 외는 접미 제거 후 매핑
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

# ----------------------------- CSV 로더 -----------------------------
NEEDED_VIS_COLS = ["광역지자체명", "기초지자체 방문자 수"]
NEEDED_PLP_COLS = ["지역", "대분류", "중분류", "대분류 지출액 비율", "중분류 지출액 비율"]

@st.cache_data(show_spinner=False)
def read_csv_forgiving(path, usecols=None, dtype=None):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, dtype=dtype)
        except Exception:
            continue
    return pd.read_csv(path, usecols=usecols, dtype=dtype)

@st.cache_data(show_spinner=False)
def read_data():
    vis = read_csv_forgiving(file_visitors, usecols=NEEDED_VIS_COLS, dtype={"광역지자체명": "string"})
    cat = read_csv_forgiving(file_spend_cat, usecols=NEEDED_PLP_COLS,
                             dtype={"지역": "string", "대분류": "string", "중분류": "string"})
    typ = read_csv_forgiving(file_spend_type, usecols=NEEDED_PLP_COLS,
                             dtype={"지역": "string", "대분류": "string", "중분류": "string"})
    for df in (cat, typ):
        for c in ["지역", "대분류", "중분류"]:
            if c in df.columns:
                df[c] = df[c].astype("category")
    return vis, cat, typ

# ----------------------- 유틸/정규화/계산 -----------------------
def normalize_region_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\(.*?\)", "", str(s))
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s).strip()
    for a in ["특별자치도","특별자치시","특별시","광역시","자치도","자치시","도","시"]:
        s = s.replace(a, "")
    s = re.sub(r"\s+", " ", s).strip()
    # 긴 이름을 2글자 약식으로
    s = to_twochar(s)
    return s

def compute_overall_share(df):
    df = df.copy()
    for c in ["대분류 지출액 비율", "중분류 지출액 비율"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["대분류_f"] = df["대분류 지출액 비율"].fillna(0)/100.0
    df["중분류_f"] = df["중분류 지출액 비율"].fillna(0)/100.0
    df["중분류_전체비중"] = df["대분류_f"] * df["중분류_f"]
    return df

def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s.fillna(0.0)
    fill_val = s.median(skipna=True)
    s = s.fillna(fill_val)
    d = s.max() - s.min()
    return (s - s.min()) / d if d > 0 else s * 0

BASE_WEIGHTS = dict(vis=0.30, div=0.30, lodg=0.20, act=0.20)

def nsi_with(df):
    return (BASE_WEIGHTS["vis"]  * df["방문자_점유율_norm"] +
            BASE_WEIGHTS["div"]  * df["소비_다양성_norm"] +
            BASE_WEIGHTS["lodg"] * df["숙박_비중_norm"] +
            BASE_WEIGHTS["act"]  * df["활동_다양성_norm"])

# ----------------------- 데이터 로드 & 전처리 -----------------------
try:
    vis, cat, typ = read_data()
except Exception as e:
    st.error("데이터 파일을 열 수 없습니다.\n\n" + str(e))
    st.stop()

# 방문자 집계/정규화
vis_region = (
    vis.groupby("광역지자체명", as_index=False)["기초지자체 방문자 수"]
      .sum()
      .rename(columns={"기초지자체 방문자 수": "방문자수_합계"})
)
total_visitors = max(vis_region["방문자수_합계"].sum(), 1)
vis_region["방문자_점유율"] = vis_region["방문자수_합계"] / total_visitors
vis_region["지역_norm"] = vis_region["광역지자체명"].map(normalize_region_name)

# 다양성/숙박/활동 지표
cat2 = compute_overall_share(cat)
typ2 = compute_overall_share(typ)

region_cat = []
for region, g in cat2.groupby("지역"):
    s = g["중분류_전체비중"].dropna().values
    if len(s) > 0 and s.sum() > 0:
        p = s / s.sum()
        H = -(p * np.log(p)).sum()
        Hmax = np.log(len(p))
        div = H / Hmax if Hmax > 0 else 0.0
    else:
        div = np.nan
    lodg = pd.to_numeric(g.loc[g["대분류"] == "숙박업", "대분류 지출액 비율"], errors="coerce")
    lodg_share = float(lodg.mean()) if len(lodg) else np.nan
    region_cat.append({"지역": region, "소비_다양성지수": div, "숙박_지출비중(%)": lodg_share})
region_cat = pd.DataFrame(region_cat)
region_cat["지역_norm"] = region_cat["지역"].map(normalize_region_name)

region_typ = []
for region, g in typ2.groupby("지역"):
    s = g["중분류_전체비중"].dropna().values
    if len(s) > 0 and s.sum() > 0:
        p = s / s.sum()
        H = -(p * np.log(p)).sum()
        Hmax = np.log(len(p))
        div_t = H / Hmax if Hmax > 0 else 0.0
    else:
        div_t = np.nan
    region_typ.append({"지역": region, "활동_다양성지수": div_t})
region_typ = pd.DataFrame(region_typ)
region_typ["지역_norm"] = region_typ["지역"].map(normalize_region_name)

@st.cache_data(show_spinner=False)
def compute_metrics(vis_region, region_cat, region_typ):
    metrics = (
        vis_region.merge(region_cat.drop(columns=["지역"]), on="지역_norm", how="left")
                  .merge(region_typ.drop(columns=["지역"]), on="지역_norm", how="left")
    )
    coords_df = pd.DataFrame([{"지역_norm": k, "lat": v[0], "lon": v[1]} for k, v in REGION_COORDS.items()])
    metrics_map = metrics.copy()
    metrics_map["방문자_점유율_norm"] = minmax(metrics_map["방문자_점유율"].fillna(0))
    metrics_map["소비_다양성_norm"]   = minmax(metrics_map["소비_다양성지수"].fillna(0))
    metrics_map["숙박_비중_norm"]     = minmax(metrics_map["숙박_지출비중(%)"].fillna(0))
    metrics_map["활동_다양성_norm"]   = minmax(metrics_map["활동_다양성지수"].fillna(0))
    metrics_map = metrics_map.merge(coords_df, on="지역_norm", how="left")
    metrics_map["NSI_base"] = nsi_with(metrics_map)
    return metrics_map

metrics_map = compute_metrics(vis_region, region_cat, region_typ)

# ==================== 인프라 지표(상가 폴더/ZIP) 통합 ====================
@st.cache_data(show_spinner=True)
def build_infra_from_sources(sources):
    import io, zipfile
    required = {"시도명","상권업종중분류명","상권업종소분류명","표준산업분류명"}
    dfs = []

    if sources["mode"] == "dir":
        for path in sources["paths"]:
            try:
                df = None
                for enc in ["cp949","utf-8","euc-kr","latin1"]:
                    try:
                        df = pd.read_csv(path, encoding=enc, low_memory=False)
                        break
                    except Exception:
                        df = None
                if df is None or not required.issubset(set(df.columns)): 
                    continue
                dfs.append(df[list(required)].copy())
            except Exception:
                continue
    elif sources["mode"] == "zip":
        zpath = sources["paths"][0]
        with zipfile.ZipFile(zpath, "r") as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                raw = z.read(name)
                df = None
                for enc in ["cp949","utf-8","euc-kr","latin1"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
                        break
                    except Exception:
                        df = None
                if df is None or not required.issubset(set(df.columns)):
                    continue
                dfs.append(df[list(required)].copy())

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    for c in required: df[c] = df[c].astype(str).str.strip()

    mid = df["상권업종중분류명"].astype(str)
    sub = df["상권업종소분류명"].astype(str)
    std = df["표준산업분류명"].astype(str)

    m_cafe = (sub.str.contains("카페")) | (std.str.contains("커피 전문점"))
    m_conv = (sub.str.contains("편의점")) | (std.str.contains("체인화 편의점"))
    m_hotel = sub.str.contains("호텔/리조트")
    m_motel = sub.str.contains("여관/모텔")
    m_accom_mid = mid.str.contains("숙박")
    m_pc = sub.str.contains("PC방")
    m_laundry = sub.str.contains("세탁소")
    m_pharmacy = sub.str.contains("약국")
    m_clinic = mid.str.contains("의원")
    m_hospital = mid.str.contains("병원") | m_clinic | sub.str.contains("치과의원|한의원")
    m_library = mid.str.contains("도서관·사적지")

    df["sido_norm"] = df["시도명"].map(lambda x: normalize_region_name(x))

    agg = df.groupby("sido_norm").agg(
        total_places=("시도명", "size"),
        cafe_count=("시도명", lambda s: int(m_cafe.loc[s.index].sum())),
        convenience_count=("시도명", lambda s: int(m_conv.loc[s.index].sum())),
        accommodation_count=("시도명", lambda s: int((m_hotel|m_motel|m_accom_mid).loc[s.index].sum())),
        hospital_count=("시도명", lambda s: int(m_hospital.loc[s.index].sum())),
        pharmacy_count=("시도명", lambda s: int(m_pharmacy.loc[s.index].sum())),
        pc_cafe_count=("시도명", lambda s: int(m_pc.loc[s.index].sum())),
        laundry_count=("시도명", lambda s: int(m_laundry.loc[s.index].sum())),
        library_museum_count=("시도명", lambda s: int(m_library.loc[s.index].sum())),
    ).reset_index()

    def per_10k(n, total):
        total = total.replace(0, np.nan)
        return (n / total) * 10000

    for col in [
        "cafe_count","convenience_count","accommodation_count","hospital_count",
        "pharmacy_count","pc_cafe_count","laundry_count","library_museum_count"
    ]:
        agg[col+"_per10k"] = per_10k(agg[col].astype(float), agg["total_places"].astype(float)).round(3)
        v = agg[col].astype(float)
        rng = v.max() - v.min()
        agg[col+"_norm"] = ((v - v.min())/rng).fillna(0).round(4) if rng > 0 else (v*0)
    return agg

infra_sources = resolve_infra_sources()
infra_df = build_infra_from_sources(infra_sources)
if not infra_df.empty:
    metrics_map = metrics_map.merge(
        infra_df.add_prefix("infra__").rename(columns={"infra__sido_norm":"지역_norm"}),
        on="지역_norm", how="left"
    )

# ============================ UI 레이아웃 ============================
st.title("디지털 노마드 지역 추천 대시보드")
left, right = st.columns([2, 1])
with left:
    st.subheader("지도에서 지역을 선택하세요")

# -------- 사이드바(업로더 제거, 기존 커뮤니티/차트 유지) --------
st.sidebar.header("추천 카테고리 선택")
cb_popular  = st.sidebar.checkbox("🔥 현재 인기 지역", value=False)
cb_toprank  = st.sidebar.checkbox("🏅 상위 랭킹 지역", value=True)
cb_hidden   = st.sidebar.checkbox("💎 숨은 보석 지역", value=False)
cb_act_rich = st.sidebar.checkbox("🎯 활동이 다양한 지역", value=False)
cb_lodging  = st.sidebar.checkbox("🛏️ 숙박이 좋은 지역", value=False)
cb_diverse  = st.sidebar.checkbox("🛍️ 소비가 다양한 지역", value=False)

st.sidebar.markdown("---")
# (향후 실제지표 연결 전까지 플레이스홀더 유지)
cb_budget   = st.sidebar.checkbox("💰 저렴한 비용(플레이스홀더)", value=False)
cb_fastnet  = st.sidebar.checkbox("🚀 빠른 인터넷(플레이스홀더)", value=False)
cb_cleanair = st.sidebar.checkbox("💨 깨끗한 공기(플레이스홀더)", value=False)
cb_safe     = st.sidebar.checkbox("🛡️ 안전한 지역(플레이스홀더)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("인프라 보너스(상가 폴더/ZIP 기반)")
cb_infra_cafe   = st.sidebar.checkbox("☕ 카페 인프라", value=False)
cb_infra_conv   = st.sidebar.checkbox("🏪 편의점 밀도", value=False)
cb_infra_accom  = st.sidebar.checkbox("🏨 숙박 시설", value=False)
cb_infra_hosp   = st.sidebar.checkbox("🏥 병원·의원", value=False)
cb_infra_pharm  = st.sidebar.checkbox("💊 약국", value=False)
cb_infra_pc     = st.sidebar.checkbox("🖥️ PC방", value=False)
cb_infra_laundry= st.sidebar.checkbox("🧺 세탁 인프라", value=False)
cb_infra_lib    = st.sidebar.checkbox("🏛️ 도서관·사적지", value=False)

st.sidebar.markdown("---")
match_mode      = st.sidebar.radio("필터 결합 방식", ["ANY(하나 이상 충족)", "ALL(모두 충족)"], index=0)
filter_strength = st.sidebar.slider("필터 강도", 0.0, 1.0, 0.5, 0.05)
bonus_strength  = st.sidebar.slider("보너스 가중", 0.0, 0.5, 0.20, 0.01)

# ----------------------------- 룰 적용 -----------------------------
def apply_category_rules(df):
    g0 = df.copy()
    notes = []

    q_vis_hi = g0["방문자_점유율_norm"].quantile(0.70)
    q_vis_lo = g0["방문자_점유율_norm"].quantile(0.40)
    q_nsi_hi = g0["NSI_base"].quantile(0.70)
    q_div_hi = g0["소비_다양성_norm"].quantile(0.70)
    q_lod_hi = g0["숙박_비중_norm"].quantile(0.70)
    q_act_hi = g0["활동_다양성_norm"].quantile(0.70)

    conds = []
    if cb_popular:  conds.append(g0["방문자_점유율_norm"] >= (q_vis_hi * (1 - 0.3 * filter_strength))); notes.append("현재 인기: 방문자 상위")
    if cb_toprank:  conds.append(g0["NSI_base"]         >= (q_nsi_hi * (1 - 0.3 * filter_strength))); notes.append("상위 랭킹: NSI 상위")
    if cb_hidden:   conds.append((g0["방문자_점유율_norm"] <= (q_vis_lo * (1 + 0.3 * filter_strength))) & (g0["NSI_base"] >= q_nsi_hi)); notes.append("숨은 보석: 방문자 하위 & NSI 상위")
    if cb_act_rich: conds.append(g0["활동_다양성_norm"] >= (q_act_hi * (1 - 0.3 * filter_strength))); notes.append("활동 다양: 활동 다양성 상위")
    if cb_lodging:  conds.append(g0["숙박_비중_norm"]   >= (q_lod_hi * (1 - 0.3 * filter_strength))); notes.append("숙박 인프라: 숙박 비중 상위")
    if cb_diverse:  conds.append(g0["소비_다양성_norm"] >= (q_div_hi * (1 - 0.3 * filter_strength))); notes.append("소비 다양: 소비 다양성 상위")

    if len(conds) == 0:
        g = g0.copy()
    else:
        if "ALL" in match_mode:
            mask = np.logical_and.reduce(conds)
            g = g0.loc[mask].copy()
            if g.empty:
                mask = np.logical_or.reduce(conds)
                g = g0.loc[mask].copy()
                notes.append("⚠️ ALL→결과 없음 → ANY로 완화")
        else:
            mask = np.logical_or.reduce(conds)
            g = g0.loc[mask].copy()
            if g.empty:
                g = g0.copy()
                notes.append("⚠️ 결과 없음 → 필터 해제")

    bonus = np.zeros(len(g))
    def add_bonus(series, strong_q): return bonus_strength * (series - strong_q).clip(lower=0)

    if cb_popular:  bonus += add_bonus(g["방문자_점유율_norm"], q_vis_hi)
    if cb_toprank:  bonus += add_bonus(g["NSI_base"], q_nsi_hi)
    if cb_act_rich: bonus += add_bonus(g["활동_다양성_norm"], q_act_hi)
    if cb_lodging:  bonus += add_bonus(g["숙박_비중_norm"], q_lod_hi)
    if cb_diverse:  bonus += add_bonus(g["소비_다양성_norm"], q_div_hi)

    def has(col): return col in g.columns and pd.api.types.is_numeric_dtype(g[col])
    infra_cols = {
        "cafe": "infra__cafe_count_norm",
        "conv": "infra__convenience_count_norm",
        "accom": "infra__accommodation_count_norm",
        "hosp": "infra__hospital_count_norm",
        "pharm": "infra__pharmacy_count_norm",
        "pc": "infra__pc_cafe_count_norm",
        "laundry": "infra__laundry_count_norm",
        "lib": "infra__library_museum_count_norm",
    }
    if cb_infra_cafe   and has(infra_cols["cafe"]):     bonus += bonus_strength * g[infra_cols["cafe"]].fillna(0)
    if cb_infra_conv   and has(infra_cols["conv"]):     bonus += bonus_strength * g[infra_cols["conv"]].fillna(0)
    if cb_infra_accom  and has(infra_cols["accom"]):    bonus += bonus_strength * g[infra_cols["accom"]].fillna(0)
    if cb_infra_hosp   and has(infra_cols["hosp"]):     bonus += bonus_strength * g[infra_cols["hosp"]].fillna(0)
    if cb_infra_pharm  and has(infra_cols["pharm"]):    bonus += bonus_strength * g[infra_cols["pharm"]].fillna(0)
    if cb_infra_pc     and has(infra_cols["pc"]):       bonus += bonus_strength * g[infra_cols["pc"]].fillna(0)
    if cb_infra_laundry and has(infra_cols["laundry"]): bonus += bonus_strength * g[infra_cols["laundry"]].fillna(0)
    if cb_infra_lib    and has(infra_cols["lib"]):      bonus += bonus_strength * g[infra_cols["lib"]].fillna(0)

    # 데모용 placeholder
    if cb_budget and has("cost_index"):
        rng = (g["cost_index"].max() - g["cost_index"].min()) + 1e-9
        tmp = 1 - ((g["cost_index"] - g["cost_index"].min()) / rng)
        bonus += bonus_strength * tmp
    if cb_fastnet and has("internet_mbps"):
        rng = (g["internet_mbps"].max() - g["internet_mbps"].min()) + 1e-9
        tmp = (g["internet_mbps"] - g["internet_mbps"].min()) / rng
        bonus += bonus_strength * tmp
    if cb_cleanair and has("air_quality_pm25"):
        rng = (g["air_quality_pm25"].max() - g["air_quality_pm25"].min()) + 1e-9
        tmp = 1 - ((g["air_quality_pm25"] - g["air_quality_pm25"].min()) / rng)
        bonus += bonus_strength * tmp
    if cb_safe and has("safety_index"):
        rng = (g["safety_index"].max() - g["safety_index"].min()) + 1e-9
        tmp = (g["safety_index"] - g["safety_index"].min()) / rng
        bonus += bonus_strength * tmp

    g["NSI"] = (g["NSI_base"] + bonus).clip(0, 1)
    return g, notes

# ----------------------------- 랭킹 계산 -----------------------------
metrics_after_rules, applied_notes = apply_category_rules(metrics_map)

if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
sel_region    = st.session_state.selected_region
selected_norm = normalize_region_name(sel_region) if sel_region else None

ranked = metrics_after_rules.copy()
ranked["NSI"]  = ranked["NSI"].fillna(ranked["NSI_base"]).fillna(0.0).clip(0,1)
ranked["rank"] = ranked["NSI"].rank(ascending=False, method="min").astype(int)

# 색: 1·2·3등 전용 + 일반
COLOR_TOP1, COLOR_TOP2, COLOR_TOP3 = "#e60049", "#ffd43b", "#4dabf7"
COLOR_SEL, COLOR_BASE = "#51cf66", "#cfd4da"

def pick_color(region_norm, selected_region_norm=None):
    if selected_region_norm and region_norm == selected_region_norm:
        return COLOR_SEL
    r = int(ranked.loc[ranked["지역_norm"] == region_norm, "rank"].min()) if region_norm in ranked["지역_norm"].values else 999
    return {1: COLOR_TOP1, 2: COLOR_TOP2, 3: COLOR_TOP3}.get(r, COLOR_BASE)

# =============================== 지도(광역만) ===============================
MAP_HEIGHT = 680
with left:
    m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles="cartodbpositron", prefer_canvas=True)

    gj, gj_err = (None, "no_path")
    if KOREA_GEOJSON and os.path.exists(KOREA_GEOJSON):
        gj, gj_err = load_geojson_safe(KOREA_GEOJSON)

    # 좌표(중심점) 준비
    coords_df = pd.DataFrame([{"지역_norm": k, "lat": v[0], "lon": v[1]} for k, v in REGION_COORDS.items()])
    ranked = ranked.drop(columns=[c for c in ["lat","lon"] if c in ranked.columns]) \
                   .merge(coords_df, on="지역_norm", how="left")

    rank_lookup = ranked.set_index("지역_norm")[["rank","NSI"]].to_dict("index")

    if gj is not None:
        # GeoJSON 속성에 2글자 REGION_NAME + 랭킹/NSI 주입
        for ft in gj.get("features", []):
            props = ft.get("properties", {})
            region_raw = None
            for k in GEO_PROP_KEYS:
                if k in props and props[k]:
                    region_raw = props[k]; break
            if region_raw is None:
                textish = [str(v) for v in props.values() if isinstance(v, str)]
                region_raw = max(textish, key=len) if textish else ""
            rname = normalize_region_name(region_raw)      # ← 2글자 약식으로 통일
            props["REGION_NAME"] = rname
            stats = rank_lookup.get(rname)
            if stats:
                props["RANK_TXT"] = f"{int(stats['rank'])}위"
                props["NSI_TXT"]  = f"{float(stats['NSI']):.3f}"
            else:
                props["RANK_TXT"] = "-"
                props["NSI_TXT"]  = "-"
            ft["properties"] = props

        # 전체 면을 랭크 색으로 채움(테두리 X, 면 채움 O)
        def style_function(feature):
            rname = feature["properties"].get("REGION_NAME", "")
            color = pick_color(rname, selected_norm)
            return {
                "fillColor": color, "color": color,
                "weight": 1, "fillOpacity": 0.70, "opacity": 0.9
            }

        def highlight_function(feature):
            return {"fillOpacity": 0.9, "weight": 2}

        GeoJson(
            gj,
            name="regions",
            style_function=style_function,
            highlight_function=highlight_function,
            smooth_factor=1.0,
            tooltip=GeoJsonTooltip(
                fields=["REGION_NAME", "RANK_TXT", "NSI_TXT"],
                aliases=["지역", "랭킹", "NSI"],
                labels=True, sticky=True,
                style=("background-color: rgba(32,32,32,0.90); color: #fff; "
                       "font-size: 12px; padding: 6px 8px; border-radius: 6px;"
                       "white-space: nowrap;"),  # 줄바꿈 방지
            ),
            popup=GeoJsonPopup(fields=["REGION_NAME"], labels=False),
        ).add_to(m)

        # 1·2·3등 범례(면 색 기준)
        legend_html = f"""
        <div style="
          position: fixed; bottom: 20px; left: 20px; z-index: 9999;
          background: rgba(255,255,255,0.96); padding: 12px 14px; border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-size: 13px; color: #222;">
          <div style="font-weight:700; margin-bottom:8px;">랭킹 범례</div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP1};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>1위</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP2};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>2위</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP3};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>3위</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_BASE};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>그 외</span>
          </div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        # GeoJSON 없으면 포인트 마커(면 채움 불가)
        for _, r in ranked.iterrows():
            if pd.isna(r.get("lat")) or pd.isna(r.get("lon")):
                continue
            color = pick_color(r["지역_norm"], selected_norm)
            nsi = float(r.get("NSI", r.get("NSI_base", 0.0))) if pd.notna(r.get("NSI", np.nan)) else 0.0
            nsi = max(min(nsi, 1.0), 0.0)
            size = 6 + 14 * nsi
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=size, color=color, fill=True, fill_color=color,
                fill_opacity=0.75, opacity=0.9, weight=1,
                popup=r["지역_norm"],
                tooltip=f"{r['지역_norm']} · {int(r['rank'])}위 · NSI {nsi:.3f}",
            ).add_to(m)

    # 클릭한 지역 세션 반영
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
    prev_clicked = st.session_state.get("_last_clicked")
    if clicked_name and clicked_name != prev_clicked:
        st.session_state.selected_region = clicked_name
        st.session_state._last_clicked = clicked_name

# ============================ 우측 패널(커뮤니티 유지) ============================
with right:
    st.subheader("커뮤니티")
    role_col1, role_col2 = st.columns(2)
    with role_col1:
        buddy_on = st.toggle("🧑‍🤝‍🧑 버디 선택", value=False)
    with role_col2:
        tourist_on = st.toggle("🧳 관광객 선택", value=False)
    st.caption(f"- 버디: **{'참여' if buddy_on else '미참여'}**  |  관광객: **{'참여' if tourist_on else '미참여'}**")

    st.markdown("### 지역 하이라이트")
    def region_reasons(row, q):
        msgs = []
        if row["방문자_점유율_norm"] >= q["vis_hi"]: msgs.append("방문 수요가 높아요")
        if row["소비_다양성_norm"]   >= q["div_hi"]: msgs.append("소비 카테고리가 다양해요")
        if row["숙박_비중_norm"]     >= q["lod_hi"]: msgs.append("숙박 인프라가 잘 갖춰져요")
        if row["활동_다양성_norm"]   >= q["act_hi"]: msgs.append("체험·활동 옵션이 풍부해요")
        if not msgs:
            best = []
            for k, lab in [("방문자_점유율_norm","방문 수요"),
                           ("소비_다양성_norm","소비 다양성"),
                           ("숙박_비중_norm","숙박 인프라"),
                           ("활동_다양성_norm","활동 다양성")]:
                best.append((row[k], lab))
            best = sorted(best, key=lambda x: x[0], reverse=True)[:2]
            msgs = [f"{lab} 상대적으로 우수" for _, lab in best]
        return " · ".join(msgs)

    q = {
        "vis_hi": ranked["방문자_점유율_norm"].quantile(0.70),
        "div_hi": ranked["소비_다양성_norm"].quantile(0.70),
        "lod_hi": ranked["숙박_비중_norm"].quantile(0.70),
        "act_hi": ranked["활동_다양성_norm"].quantile(0.70),
    }
    top_show = ranked.sort_values("NSI", ascending=False).head(5)
    for _, r in top_show.iterrows():
        st.write(f"**{r['지역_norm']}** — {int(r['rank'])}위 · NSI {float(r['NSI']):.3f}")
        st.caption("· " + region_reasons(r, q))

    # QnA · 게시판 (기존 유지)
    st.markdown("### QnA · 게시판")
    STORE_PATH = os.path.join(DATA_DIR, "community_qna.json")

    def load_store():
        try:
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"posts": []}

    def save_store(data):
        try:
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if "qna_store" not in st.session_state:
        st.session_state.qna_store = load_store()
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
                "id": str(uuid.uuid4()), "type": "qna",
                "title": title.strip(), "content": content.strip(),
                "region": normalize_region_name(region_tag) if region_tag else "",
                "author": "익명", "created": int(time.time()), "answers": []
            })
            save_store(store); st.success("질문이 등록되었습니다.")
    with tabs[1]:
        with st.form("form_board"):
            title2 = st.text_input("제목 ", value="")
            content2 = st.text_area("본문", height=140, value="")
            region_tag2 = st.text_input("지역 태그(선택)", value=st.session_state.selected_region or "")
            submit2 = st.form_submit_button("글 등록")
        if submit2 and title2.strip():
            store["posts"].append({
                "id": str(uuid.uuid4()), "type": "board",
                "title": title2.strip(), "content": content2.strip(),
                "region": normalize_region_name(region_tag2) if region_tag2 else "",
                "author": "익명", "created": int(time.time()), "comments": []
            })
            save_store(store); st.success("글이 등록되었습니다.")
    with tabs[2]:
        colf1, colf2 = st.columns([1,1])
        with colf1:
            feed_type = st.multiselect("유형", ["qna", "board"], default=["qna", "board"])
        with colf2:
            feed_region = st.text_input("지역 필터(부분일치)", value="")
        posts = [p for p in store["posts"] if p["type"] in feed_type]
        if feed_region.strip():
            key = normalize_region_name(feed_region)
            posts = [p for p in posts if key in normalize_region_name(p.get("region",""))]
        posts = sorted(posts, key=lambda p: p.get("created", 0), reverse=True)
        for p in posts:
            with st.expander(f"[{'QnA' if p['type']=='qna' else '게시글'}] {p['title']}  ·  {p.get('region','') or '전체'}"):
                st.write(p["content"] or "(내용 없음)")
                if p["type"] == "qna":
                    for a in p.get("answers", []):
                        st.markdown(f"- **답변**: {a['content']}  — _{a.get('author','익명')}_")
                    with st.form(f"ans_{p['id']}"):
                        ans = st.text_input("답변 달기", value="")
                        ans_btn = st.form_submit_button("등록")
                    if ans_btn and ans.strip():
                        p.setdefault("answers", []).append({"content": ans.strip(), "author": "익명", "created": int(time.time())})
                        save_store(store); st.success("답변이 등록되었습니다.")
                else:
                    for cmt in p.get("comments", []):
                        st.markdown(f"- **댓글**: {cmt['content']}  — _{cmt.get('author','익명')}_")
                    with st.form(f"cmt_{p['id']}"):
                        cmt = st.text_input("댓글 달기", value="")
                        cmt_btn = st.form_submit_button("등록")
                    if cmt_btn and cmt.strip():
                        p.setdefault("comments", []).append({"content": cmt.strip(), "author": "익명", "created": int(time.time())})
                        save_store(store); st.success("댓글이 등록되었습니다.")

# ============================ 랭킹/키워드 ============================
st.subheader("추천 랭킹")
cols_to_show = ["광역지자체명","NSI","NSI_base","방문자수_합계","방문자_점유율",
     "숙박_지출비중(%)","소비_다양성지수","활동_다양성지수"]
if not infra_df.empty:
    cols_to_show += [
        "infra__cafe_count_per10k","infra__convenience_count_per10k","infra__accommodation_count_per10k",
        "infra__hospital_count_per10k","infra__pharmacy_count_per10k"
    ]
rec = ranked.sort_values("NSI", ascending=False)[cols_to_show]

out = rec.copy()
if "방문자수_합계" in out.columns:
    out["방문자수_합계"] = out["방문자수_합계"].fillna(0).astype(int)
for c in out.columns:
    if c not in ["광역지자체명"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

st.dataframe(rec.reset_index(drop=True), use_container_width=True)
st.download_button(
    "⬇️ 랭킹 CSV 저장",
    out.to_csv(index=False).encode("utf-8-sig"),
    file_name="ranking_by_categories_and_infra.csv", mime="text/csv"
)

st.subheader("키워드 · 카테고리 탐색 (업종/유형 검색비중 기반)")
def safe_categories(series):
    try:
        if hasattr(series, "cat"):
            vals = list(series.cat.categories)
        else:
            vals = pd.Series(series, dtype="string").dropna().unique().tolist()
        vals = [v for v in map(str, vals) if v.strip()]
    except Exception:
        vals = []
    return ["--전체--"] + sorted(vals)

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input("지역", st.session_state.selected_region or "", disabled=True)
with col2:
    sel_big = st.selectbox("대분류 선택(업종)", safe_categories(cat["대분류"]))
with col3:
    kw = st.text_input("키워드(중분류명 부분 일치)", "")

def top_keywords(df, region, big=None, keyword="", topn=10):
    region_normed = normalize_region_name(region or "")
    g = df[df["지역"].astype(str).map(normalize_region_name) == region_normed].copy()
    if big and big != "--전체--":
        g = g[g["대분류"] == big]
    if keyword:
        g = g[g["중분류"].astype(str).str.contains(keyword, case=False, na=False)]
    g["중분류_전체비중"] = (
        pd.to_numeric(g["대분류 지출액 비율"], errors="coerce")/100.0 *
        pd.to_numeric(g["중분류 지출액 비율"], errors="coerce")/100.0
    )
    g = g.groupby(["대분류","중분류"], as_index=False)["중분류_전체비중"].sum() \
         .sort_values("중분류_전체비중", ascending=False).head(topn)
    return g

if st.session_state.selected_region:
    tabs2 = st.tabs(["업종 기준(PLP_업종별)", "유형 기준(PLP_유형별)"])
    with tabs2[0]:
        top_cat = top_keywords(cat, st.session_state.selected_region, sel_big, kw, topn=12)
        st.dataframe(top_cat, use_container_width=True)
        st.bar_chart(top_cat.set_index("중분류")["중분류_전체비중"])
    with tabs2[1]:
        sel_big2 = st.selectbox("대분류 선택(유형)", safe_categories(typ["대분류"]), key="type_big")
        kw2 = st.text_input("키워드(중분류명 부분 일치)", "", key="type_kw")
        top_typ = top_keywords(typ, st.session_state.selected_region, sel_big2, kw2, topn=12)
        st.dataframe(top_typ, use_container_width=True)
        st.bar_chart(top_typ.set_index("중분류")["중분류_전체비중"])
else:
    st.info("지도에서 영역을 클릭하거나, 우측 패널·드롭다운을 이용해 지역을 선택하세요.")

st.markdown("""
---
**데이터 출처**  
- 관광 데이터랩: 광역별 방문자 수  
- PLP 데이터: 업종/유형별 지출비중(검색 비중 기반)  
- 소상공인시장진흥공단: 상가(상권)정보 (폴더/ZIP, 시도 인프라 집계)
""")
