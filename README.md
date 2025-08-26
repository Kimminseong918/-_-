# -_-
한국관광데이티랩 공모전

# Nomad 지역 추천 대시보드

# 디지털 노마드 지역 추천 대시보드 – 방법론

본 서비스는 디지털 노마드가 **장기 체류하기 좋은 지역**을 찾을 수 있도록  
방문 수요, 생활 인프라, 숙박 기반, 활동 매력도를 종합한 **Nomad Suitability Index (NSI)**를 설계·적용합니다.

## NSI 산출 플로우
입력 데이터 → 핵심 지표(방문자·소비·숙박·활동) → 정규화(0~1) → NSI 산출(가중치+보너스) → 필터/랭킹 → 지도 & 추천



## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py


소상공인시장진흥공단 상가(상권)정보 (공공데이터포털)

데이터 다운로드 페이지: https://www.data.go.kr/data/15083033/fileData.do#/tab-layer-openapi
제공기관: 소상공인시장진흥공단
파일명: 소상공인시장진흥공단_상가(상권)정보_20250630.zip
