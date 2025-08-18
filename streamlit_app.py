# -*- coding: utf-8 -*-
import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# 지도
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
from streamlit_folium import st_folium

st.set_page_config(page_title="Nomad 추천 대시보드", layout="wide")

# =========================================================
# 경로 후보
# =========================================================
CANDIDATE_BASES = [
    r"C:\Users\123cl\OneDrive\바탕 화면\test",  # Windows OneDrive
    "/mnt/data",                                # 리눅스/노트북
    ".",                                        # 현재 폴더
]

def build_paths():
    for base in CANDIDATE_BASES:
        fv = os.path.join(base, "20250809144224_광역별 방문자 수.csv")
        fc = os.path.join(base, "PLP_업종별_검색건수_통합.csv")
        ft = os.path.join(base, "PLP_유형별_검색건수_통합.csv")
        if all(os.path.exists(p) for p in [fv, fc, ft]):
            return fv, fc, ft
    return fv, fc, ft

file_visitors, file_spend_cat, file_spend_type = build_paths()

def resolve_geojson_path():
    candidates = [
        os.path.join(CANDIDATE_BASES[0], "korea_provinces.geojson"),
        os.path.join(CANDIDATE_BASES[0], "KOREA_GEOJSON.geojson"),
        "/mnt/data/korea_provinces.geojson",
        "/mnt/data/KOREA_GEOJSON.geojson",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

KOREA_GEOJSON = resolve_geojson_path()

# 국내 GeoJSON의 지역명 키 후보 (KOSTAT 2018 호환: name 우선)
GEO_PROP_KEYS = ["name", "CTPRVN_NM", "ADM1_KOR_NM", "sido_nm", "SIG_KOR_NM", "NAME_1"]

# =========================================================
# 안전 GeoJSON 로더 + 캐시
# =========================================================
@st.cache_data(show_spinner=False)
def load_geojson_safe(path: str):
    import re
    if not path or not os.path.exists(path):
        return None, "missing_path"
    try:
        if os.path.getsize(path) == 0:
            return None, "empty_file"
    except Exception:
        pass

    encodings = (
        "utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be",
        "cp949", "euc-kr", "latin-1"
    )
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                txt = f.read().strip()
            if not txt:
                return None, f"empty_text({enc})"
            if txt.lstrip().startswith(("var ", "const ", "let ")):  # JS 변수 래핑
                m = re.search(r"\{.*\}\s*;?\s*$", txt, flags=re.S)
                if m:
                    txt = m.group(0)
            gj = json.loads(txt)
            if not isinstance(gj, dict):
                return None, f"not_dict({enc})"
            if gj.get("type") == "Topology":
                return None, "topojson_not_supported"
            if gj.get("type") != "FeatureCollection" or not isinstance(gj.get("features"), list):
                return None, f"not_featurecollection({enc})"
            return gj, None
        except Exception as e:
            last_err = f"{enc}: {e}"
            continue
    return None, last_err or "unknown_error"

# === 지역(광역) 중심점 좌표 (대략값) ===
REGION_COORDS = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756), "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052), "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890), "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555), "충북": (36.6357, 127.4913), "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530), "전남": (34.8679, 126.9910), "경북": (36.4919, 128.8889),
    "경남": (35.4606, 128.2132), "제주": (33.4996, 126.5312),
}

# ---------- CSV 로더(최적화: 필요한 컬럼만, 캐시) ----------
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
    # 문자열 압축
    for df in (cat, typ):
        for c in ["지역", "대분류", "중분류"]:
            if c in df.columns:
                df[c] = df[c].astype("category")
    return vis, cat, typ

# =============== 데이터 로드 ===============
try:
    vis, cat, typ = read_data()
except Exception as e:
    st.error(
        f"데이터 파일을 찾거나 열 수 없습니다.\n\n"
        f"- 방문자 수: {file_visitors}\n- 업종별: {file_spend_cat}\n- 유형별: {file_spend_type}\n\n{e}"
    )
    st.stop()

# =============== 전처리/지표 계산 ===============
def normalize_region_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for a in ["특별자치도","특별자치시","특별시","광역시","자치도","자치시","도","시"]:
        s = s.replace(a, "")
    return " ".join(s.split())

def compute_overall_share(df):
    df = df.copy()
    for c in ["대분류 지출액 비율", "중분류 지출액 비율"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["대분류_f"] = df["대분류 지출액 비율"] / 100.0
    df["중분류_f"] = df["중분류 지출액 비율"] / 100.0
    df["중분류_전체비중"] = df["대분류_f"] * df["중분류_f"]
    return df

# 방문자 점유율
vis_region = (
    vis.groupby("광역지자체명", as_index=False)["기초지자체 방문자 수"]
      .sum()
      .rename(columns={"기초지자체 방문자 수": "방문자수_합계"})
)
total_visitors = max(vis_region["방문자수_합계"].sum(), 1)
vis_region["방문자_점유율"] = vis_region["방문자수_합계"] / total_visitors
vis_region["지역_norm"] = vis_region["광역지자체명"].map(normalize_region_name)

# 소비/유형 다양성 & 숙박 비중
cat2 = compute_overall_share(cat)
typ2 = compute_overall_share(typ)

# region-level metrics (cat2)
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

# region-level metrics (typ2)
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

# 통합 메트릭
metrics = (
    vis_region.merge(region_cat.drop(columns=["지역"]), on="지역_norm", how="left")
              .merge(region_typ.drop(columns=["지역"]), on="지역_norm", how="left")
)

def minmax(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

metrics["방문자_점유율_norm"] = minmax(metrics["방문자_점유율"].fillna(0))
metrics["소비_다양성_norm"]   = minmax(metrics["소비_다양성지수"].fillna(0))
metrics["숙박_비중_norm"]     = minmax(metrics["숙박_지출비중(%)"].fillna(0))
metrics["활동_다양성_norm"]   = minmax(metrics["활동_다양성지수"].fillna(0))

# 기본 NSI(체크박스 적용 전 기본 가중치)
BASE_WEIGHTS = dict(vis=0.30, div=0.30, lodg=0.20, act=0.20)
def base_nsi(df):
    return (
        BASE_WEIGHTS["vis"]  * df["방문자_점유율_norm"] +
        BASE_WEIGHTS["div"]  * df["소비_다양성_norm"] +
        BASE_WEIGHTS["lodg"] * df["숙박_비중_norm"] +
        BASE_WEIGHTS["act"]  * df["활동_다양성_norm"]
    )
metrics["NSI_base"] = base_nsi(metrics)

# 좌표 DF
coords_df = pd.DataFrame([{"지역_norm": k, "lat": v[0], "lon": v[1]} for k, v in REGION_COORDS.items()])
metrics_map = metrics.merge(coords_df, on="지역_norm", how="left")

# =====================================================
# UI (좌: 지도 / 우: 커뮤니티 패널)
# =====================================================
st.title("디지털 노마드 지역 추천 대시보드")

left, right = st.columns([2, 1])

with left:
    st.subheader("지도에서 지역을 선택하세요")

# 사이드바 필터들 ------------------------------------------------
st.sidebar.header("추천 카테고리 선택")
st.sidebar.caption("여러 조건을 선택하세요. (필터 + 보너스 가중)")
colA1, colA2 = st.sidebar.columns(2)
with colA1:
    cb_popular = st.checkbox("🔥 Popular now", value=False)
    cb_toprank = st.checkbox("🏅 Top ranked", value=True)
    cb_hidden  = st.checkbox("💎 Hidden gem", value=False)
with colA2:
    cb_act_rich = st.checkbox("🎯 Activity-rich", value=False)
    cb_lodging  = st.checkbox("🛏️ Lodging-ready", value=False)
    cb_diverse  = st.checkbox("🛍️ Diverse consumption", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("추가 지표(해당 컬럼이 있을 때 자동 반영)")
colB1, colB2 = st.sidebar.columns(2)
with colB1:
    cb_budget  = st.checkbox("💰 Budget-friendly", value=False)
    cb_fastnet = st.checkbox("🚀 Fast internet", value=False)
with colB2:
    cb_cleanair = st.checkbox("💨 Clean air now", value=False)
    cb_safe     = st.checkbox("🛡️ Safe", value=False)

st.sidebar.markdown("---")
match_mode = st.sidebar.radio("필터 결합 방식", ["ANY(하나 이상 충족)", "ALL(모두 충족)"], index=0)
filter_strength = st.sidebar.slider("필터 강도", 0.0, 1.0, 0.5, 0.05)
bonus_strength  = st.sidebar.slider("보너스 가중", 0.0, 0.5, 0.20, 0.01)

# --------------------------------------------------------------
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
    if cb_popular:
        conds.append(g0["방문자_점유율_norm"] >= (q_vis_hi * (1 - 0.3 * filter_strength)))
        notes.append("Popular now: 방문자 상위")
    if cb_toprank:
        conds.append(g0["NSI_base"] >= (q_nsi_hi * (1 - 0.3 * filter_strength)))
        notes.append("Top ranked: NSI 상위")
    if cb_hidden:
        conds.append((g0["방문자_점유율_norm"] <= (q_vis_lo * (1 + 0.3 * filter_strength))) & (g0["NSI_base"] >= q_nsi_hi))
        notes.append("Hidden gem: 방문자 하위 & NSI 상위")
    if cb_act_rich:
        conds.append(g0["활동_다양성_norm"] >= (q_act_hi * (1 - 0.3 * filter_strength)))
        notes.append("Activity-rich: 활동 다양성 상위")
    if cb_lodging:
        conds.append(g0["숙박_비중_norm"] >= (q_lod_hi * (1 - 0.3 * filter_strength)))
        notes.append("Lodging-ready: 숙박 비중 상위")
    if cb_diverse:
        conds.append(g0["소비_다양성_norm"] >= (q_div_hi * (1 - 0.3 * filter_strength)))
        notes.append("Diverse consumption: 소비 다양성 상위")

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

    if cb_popular: bonus += add_bonus(g["방문자_점유율_norm"], q_vis_hi)
    if cb_toprank: bonus += add_bonus(g["NSI_base"], q_nsi_hi)
    if cb_act_rich: bonus += add_bonus(g["활동_다양성_norm"], q_act_hi)
    if cb_lodging:  bonus += add_bonus(g["숙박_비중_norm"], q_lod_hi)
    if cb_diverse:  bonus += add_bonus(g["소비_다양성_norm"], q_div_hi)

    def has_col(col): return col in g.columns and pd.api.types.is_numeric_dtype(g[col])
    if cb_budget and has_col("cost_index"):
        rng = (g["cost_index"].max() - g["cost_index"].min()) + 1e-9
        tmp = 1 - ((g["cost_index"] - g["cost_index"].min()) / rng)
        bonus += bonus_strength * tmp
        notes.append("Budget: cost_index 적용")
    if cb_fastnet and has_col("internet_mbps"):
        rng = (g["internet_mbps"].max() - g["internet_mbps"].min()) + 1e-9
        tmp = (g["internet_mbps"] - g["internet_mbps"].min()) / rng
        bonus += bonus_strength * tmp
        notes.append("Fast net: internet_mbps 적용")
    if cb_cleanair and has_col("air_quality_pm25"):
        rng = (g["air_quality_pm25"].max() - g["air_quality_pm25"].min()) + 1e-9
        tmp = 1 - ((g["air_quality_pm25"] - g["air_quality_pm25"].min()) / rng)
        bonus += bonus_strength * tmp
        notes.append("Clean air: pm2.5 적용")
    if cb_safe and has_col("safety_index"):
        rng = (g["safety_index"].max() - g["safety_index"].min()) + 1e-9
        tmp = (g["safety_index"] - g["safety_index"].min()) / rng
        bonus += bonus_strength * tmp
        notes.append("Safe: safety_index 적용")

    g["NSI"] = g["NSI_base"] + bonus
    return g, notes

# ======== 메트릭/랭킹 생성 ========
metrics_after_rules, applied_notes = apply_category_rules(metrics_map)
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None

sel_region = st.session_state.selected_region
selected_norm = normalize_region_name(sel_region) if sel_region else None

ranked = metrics_after_rules.copy()
ranked["rank"] = ranked["NSI"].rank(ascending=False, method="min").astype(int)

# 색상
COLOR_TOP1, COLOR_TOP2, COLOR_TOP3 = "#ff6b6b", "#ffd43b", "#4dabf7"
COLOR_SEL, COLOR_BASE = "#94d82d", "#b8c0cc"

def pick_color(region_norm, selected_region_norm=None):
    if selected_region_norm and region_norm == selected_region_norm:
        return COLOR_SEL
    r = int(ranked.loc[ranked["지역_norm"] == region_norm, "rank"].min()) if region_norm in ranked["지역_norm"].values else 999
    return {1: COLOR_TOP1, 2: COLOR_TOP2, 3: COLOR_TOP3}.get(r, COLOR_BASE)

# =================== 지도(왼쪽) ===================
with left:
    m = folium.Map(
        location=[36.5, 127.8],
        zoom_start=7,
        tiles="cartodbpositron",
        prefer_canvas=True   # 렌더 최적화
    )

    gj, gj_err = (None, "no_path")
    if KOREA_GEOJSON and os.path.exists(KOREA_GEOJSON):
        gj, gj_err = load_geojson_safe(KOREA_GEOJSON)

    if gj is None:
        st.warning(f"GeoJSON 로드 실패 → 마커 모드, 원인: {gj_err}")
        # 폴리곤 실패 시 마커 모드
        for _, r in ranked.iterrows():
            if pd.isna(r.get("lat")) or pd.isna(r.get("lon")): continue
            color = pick_color(r["지역_norm"], selected_norm)
            nsi = float(r.get("NSI", r.get("NSI_base", 0.0)))
            size = 6 + 14 * nsi
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=size, color=color, fill=True, fill_color=color,
                fill_opacity=0.75, opacity=0.9 if color != COLOR_BASE else 0.6,
                weight=2 if color != COLOR_BASE else 1
            ).add_to(m)
    else:
        # REGION_NAME 주입
        for ft in gj.get("features", []):
            props = ft.get("properties", {})
            region_raw = None
            for k in GEO_PROP_KEYS:
                if k in props and props[k]:
                    region_raw = props[k]; break
            if region_raw is None:
                textish = [str(v) for v in props.values() if isinstance(v, str)]
                region_raw = max(textish, key=len) if textish else ""
            props["REGION_NAME"] = normalize_region_name(region_raw)

        def style_function(feature):
            rname = feature["properties"].get("REGION_NAME", "")
            color = pick_color(rname, selected_norm)
            return {
                "fillColor": color, "color": color,
                "weight": 2 if color != COLOR_BASE else 1,
                "fillOpacity": 0.45 if color != COLOR_BASE else 0.25,
                "opacity": 0.9 if color != COLOR_BASE else 0.6,
            }

        def highlight_function(feature):
            return {"fillOpacity": 0.75, "weight": 3}

        GeoJson(
            gj,
            name="regions",
            style_function=style_function,
            highlight_function=highlight_function,
            smooth_factor=1.0,   # 경계 간소화(렌더 가벼움)
            tooltip=GeoJsonTooltip(
                fields=["REGION_NAME"], labels=False, sticky=True,
                style=("background-color: rgba(32,32,32,0.85); color: white; "
                       "font-size: 12px; padding: 6px 8px; border-radius: 6px;"),
            ),
            popup=GeoJsonPopup(fields=["REGION_NAME"], labels=False),
        ).add_to(m)

        # 범례(1~3위)
        legend_html = f"""
        <div style="
          position: fixed; bottom: 20px; left: 20px; z-index: 9999;
          background: rgba(255,255,255,0.96); padding: 12px 14px; border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-size: 13px; color: #222;">
          <div style="font-weight:700; margin-bottom:8px;">랭킹</div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP1};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>1위 · 최상위 지역</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP2};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>2위 · 상위권 지역</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <span style="display:inline-block;width:14px;height:14px;background:{COLOR_TOP3};
                         border:1px solid rgba(0,0,0,.2);border-radius:3px;"></span>
            <span>3위 · 유망 지역</span>
          </div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    # 지도를 조금 더 작게(높이 420)
    MAP_KEY = "main_map"
    map_state = st_folium(m, width=None, height=420, key=MAP_KEY)

    # 클릭 디바운스
    clicked = map_state.get("last_object_clicked_popup")
    prev_clicked = st.session_state.get("_last_clicked")
    if clicked and clicked != prev_clicked:
        st.session_state.selected_region = normalize_region_name(str(clicked))
        st.session_state._last_clicked = clicked

# =================== 커뮤니티 패널(오른쪽) ===================
with right:
    st.subheader("커뮤니티")
    # 카드 스타일 컨테이너
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 12px;
            padding: 16px 16px 6px 16px;">
            <div style="font-weight:700; font-size:16px; margin-bottom:10px;">
                역할 선택
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    # 실제 인터랙션 위젯
    role_col1, role_col2 = st.columns(2)
    with role_col1:
        buddy_on = st.toggle("🧑‍🤝‍🧑 버디 선택", value=False, help="지역 청년/학생 버디로 참여")
    with role_col2:
        tourist_on = st.toggle("🧳 관광객 선택", value=False, help="체류/여행자로 참여")

    # 상태 표시
    st.caption("선택 상태")
    st.write(
        f"- 버디: **{'참여' if buddy_on else '미참여'}**  |  "
        f"관광객: **{'참여' if tourist_on else '미참여'}**"
    )

# =================== 이하: 랭킹/키워드 섹션 ===================
st.subheader("추천 랭킹")
rec = ranked.sort_values("NSI", ascending=False)[
    ["광역지자체명","NSI","NSI_base","방문자수_합계","방문자_점유율",
     "숙박_지출비중(%)","소비_다양성지수","활동_다양성지수"]
]
st.dataframe(rec.reset_index(drop=True), use_container_width=True)
st.download_button(
    "⬇️ 랭킹 CSV 저장",
    rec.to_csv(index=False).encode("utf-8-sig"),
    file_name="ranking_by_categories.csv",
    mime="text/csv"
)

st.subheader("키워드 · 카테고리 탐색 (업종/유형 검색비중 기반)")
col1, col2, col3 = st.columns(3)
with col1:
    st.text_input("지역", st.session_state.selected_region or "", disabled=True)
with col2:
    sel_big = st.selectbox("대분류 선택(업종)", ["--전체--"] + sorted(cat["대분류"].cat.categories.tolist()))
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
    tabs = st.tabs(["업종 기준(PLP_업종별)", "유형 기준(PLP_유형별)"])
    with tabs[0]:
        top_cat = top_keywords(cat, st.session_state.selected_region, sel_big, kw, topn=12)
        st.dataframe(top_cat, use_container_width=True)
        st.bar_chart(top_cat.set_index("중분류")["중분류_전체비중"])
    with tabs[1]:
        sel_big2 = st.selectbox("대분류 선택(유형)", ["--전체--"] + sorted(typ["대분류"].cat.categories.tolist()), key="type_big")
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
""")
