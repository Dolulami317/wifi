import os
import pandas as pd
import numpy as np
import streamlit as pd_st # streamlit 임포트
import streamlit as st
import folium
from streamlit_folium import st_folium

# 1. 파일 경로 설정 및 데이터 로드
file_name = '서울특별시 공공와이파이 AP별 사용량_고정형(20210501_20211013).csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)

@st.cache_data # 데이터를 매번 새로 로드하지 않고 캐싱하여 속도 향상
def load_and_process_data():
    df = pd.read_csv(file_path, encoding='cp949')
    df.columns = df.columns.str.strip()
    
    # 자치구 통계 계산 및 Z-score 산출
    stats = df.groupby('자치구')['AP별 이용량(GB)'].agg(['mean', 'std']).reset_index()
    stats.columns = ['자치구', '구별_평균_이용량', '구별_표준편차']
    merged = pd.merge(df, stats, on='자치구', how='left')
    merged['Z_score'] = (merged['AP별 이용량(GB)'] - merged['구별_평균_이용량']) / (merged['구별_표준편차'] + 1e-6)
    
    # 상태 판정 함수
    def detect_anomaly(z):
        if z >= 3.0:
            return '과부하/인파밀집'
        elif z <= -2.0:
            return '장비 장애 의심'
        else:
            return '정상'
            
    merged['상태'] = merged['Z_score'].apply(detect_anomaly)
    
    # 실시간 지도 시각화를 위한 가상의 위경도 추가 (실제 API 정보 결합 전 임시 데이터)
    # 서울 중심좌표(37.5665, 126.9780) 부근으로 임의 배정
    np.random.seed(42)
    merged['latitude'] = 37.5665 + np.random.uniform(-0.08, 0.08, size=len(merged))
    merged['longitude'] = 126.9780 + np.random.uniform(-0.12, 0.12, size=len(merged))
    
    return merged

df_processed = load_and_process_data()

# 2. 웹앱 UI 타이틀 및 사이드바 설정
st.set_page_config(layout="wide")
st.title("📡 서울시 공공 와이파이 실시간 장애 및 과부하 관제탑")
st.markdown("Z-score 실시간 이상탐지 알고리즘 기반 네트워크 모니터링 시스템")

# 사이드바 필터링
st.sidebar.header("조회 필터 설정")
selected_gu = st.sidebar.selectbox("조회할 자치구 선택", ['전체'] + list(df_processed['자치구'].unique()))
selected_status = st.sidebar.multiselect("와이파이 상태 필터", ['정상', '과부하/인파밀집', '장비 장애 의심'], default=['과부하/인파밀집', '장비 장애 의심'])

# 필터 적용
filtered_df = df_processed.copy()
if selected_gu != '전체':
    filtered_df = filtered_df[filtered_df['자치구'] == selected_gu]
filtered_df = filtered_df[filtered_df['상태'].isin(selected_status)]

# 3. 주요 스탯 표시 (Metric)
col1, col2, col3 = st.columns(3)
col1.metric("총 모니터링 장비 수", f"{len(df_processed):,} 개")
col2.metric("과부하/인파밀집 수", f"{len(df_processed[df_processed['상태']=='과부하/인파밀집']):,} 개", delta="과부하 주의", delta_color="inverse")
col3.metric("장애 의심 장비 수", f"{len(df_processed[df_processed['상태']=='장비 장애 의심']):,} 개", delta="점검 필요", delta_color="normal")

# 4. 지도 시각화
st.subheader("📍 서울시 공공와이파이 장애/밀집 관제 지도")
st.caption("안전과 트래픽 분산을 위해 이상징후(과부하: 빨강/노랑, 장애: 주황)가 발생한 구역을 표시합니다.")

# 지도의 중심 설정
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

# 지도에 최대 200개 마커만 표시 (브라우저 과부하 방지)
map_data = filtered_df.head(200)

for idx, row in map_data.iterrows():
    color = 'blue'
    if row['상태'] == '과부하/인파밀집':
        color = 'red'
    elif row['상태'] == '장비 장애 의심':
        color = 'orange'
        
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"AP: {row['관리번호']}<br>구: {row['자치구']}<br>사용량: {row['AP별 이용량(GB)']:.2f}GB<br>Z-score: {row['Z_score']:.2f}",
        tooltip=f"{row['관리번호']} ({row['상태']})",
        icon=folium.Icon(color=color, icon='wifi', prefix='fa')
    ).add_to(m)

# Streamlit에 지도 렌더링
st_folium(m, width=1200, height=500)

# 5. 데이터 표 출력
st.subheader("📋 이상 탐지 리스트 (상위 50개)")
st.dataframe(filtered_df[['관리번호', '자치구', 'AP별 이용량(GB)', 'Z_score', '상태']].head(50), use_container_width=True)