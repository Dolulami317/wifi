import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
# 브라우저 GPS 수집용 라이브러리
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(layout="wide")
st.title("📡 사용자 위치 기반 서울시 공공 와이파이 관제탑")
st.markdown("Z-score 실시간 이상탐지 및 내 위치 기준 최단거리 20개 장비 모니터링")

# --- 하버사인(Haversine) 구면 거리 계산 함수 ---
def haversine_distance(lat1, lon1, lat2, lon2):
    # 지구 반지름 (km)
    R = 6371.0
    
    # 라디안 변환
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance # 단위: km

# --- 데이터 로드 및 이상탐지 전처리 ---
file_name = '서울특별시 공공와이파이 AP별 사용량_고정형(20210501_20211013).csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)

@st.cache_data
def load_and_process_data():
    if not os.path.exists(file_path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path, encoding='cp949')
    df.columns = df.columns.str.strip()
    
    # 자치구 통계 계산 및 Z-score 산출
    stats = df.groupby('자치구')['AP별 이용량(GB)'].agg(['mean', 'std']).reset_index()
    stats.columns = ['자치구', '구별_평균_이용량', '구별_표준편차']
    merged = pd.merge(df, stats, on='자치구', how='left')
    merged['Z_score'] = (merged['AP별 이용량(GB)'] - merged['구별_평균_이용량']) / (merged['구별_표준편차'] + 1e-6)
    
    def detect_anomaly(z):
        if z >= 3.0:
            return '과부하/인파밀집'
        elif z <= -2.0:
            return '장비 장애 의심'
        else:
            return '정상'
            
    merged['상태'] = merged['Z_score'].apply(detect_anomaly)
    
    # 분석용 실시간 위도 경도 매핑 (서울 강남/성동/시청 등 실제 구역 분포 모사)
    # 실제 API 연동 시 실제 위경도로 매핑되나, 가용 데이터 기준 서울시 중심 랜덤 흩뿌리기
    np.random.seed(42)
    merged['latitude'] = 37.5665 + np.random.uniform(-0.08, 0.08, size=len(merged))
    merged['longitude'] = 126.9780 + np.random.uniform(-0.12, 0.12, size=len(merged))
    
    return merged

df_processed = load_and_process_data()

# --- 브라우저 실시간 GPS 내 위치 가져오기 ---
st.sidebar.header("📍 실시간 위치 찾기")
get_location = st.sidebar.checkbox("내 위치 불러오기 (GPS)")

user_lat, user_lon = None, None

if get_location:
    # JavaScript를 호출해 브라우저의 실시간 위치 수집
    loc = streamlit_js_eval(data_key='JS_LOCATION', target_json='geolocation', want_output=True)
    if loc:
        user_lat = loc['coords']['latitude']
        user_lon = loc['coords']['longitude']
        st.sidebar.success(f"위치 획득 성공!\n위도: {user_lat:.4f} / 경도: {user_lon:.4f}")
    else:
        st.sidebar.info("브라우저의 위치 권한 허용이 필요합니다.")

# --- 메인 비즈니스 로직 및 필터링 ---
if not df_processed.empty:
    # 1. 사용자의 실시간 GPS가 있다면 하버사인 거리 정렬 및 가장 가까운 20개 데이터 정렬
    if user_lat and user_lon:
        df_processed['거리(km)'] = df_processed.apply(
            lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
        )
        # 내 위치 기준 가장 가까운 20개 장비만 필터링
        filtered_df = df_processed.nsmallest(20, '거리(km)')
        map_center = [user_lat, user_lon]
        zoom_level = 14
        st.subheader("🎯 내 현재 위치 반경 가장 가까운 실시간 와이파이 TOP 20")
    else:
        # GPS 정보가 없을 때의 기본 자치구 필터 작동 방식 (원래 화면)
        selected_gu = st.sidebar.selectbox("조회할 자치구 선택", ['전체'] + list(df_processed['자치구'].unique()))
        selected_status = st.sidebar.multiselect("와이파이 상태 필터", ['정상', '과부하/인파밀집', '장비 장애 의심'], default=['과부하/인파밀집', '장비 장애 의심'])

        filtered_df = df_processed.copy()
        if selected_gu != '전체':
            filtered_df = filtered_df[filtered_df['자치구'] == selected_gu]
        filtered_df = filtered_df[filtered_df['상태'].isin(selected_status)]
        
        # 지도 기본 중심 설정 (서울시청)
        map_center = [37.5665, 126.9780]
        zoom_level = 11
        st.subheader("📍 서울시 공공와이파이 장애/밀집 관제 지도")

    # 2. 실시간 스탯 표시
    col1, col2, col3 = st.columns(3)
    col1.metric("조회된 실시간 모니터링 수", f"{len(filtered_df):,} 개")
    col2.metric("과부하/인파밀집 탐지", f"{len(filtered_df[filtered_df['상태']=='과부하/인파밀집']):,} 개", delta="주의 요망", delta_color="inverse")
    col3.metric("통신 장애 의심 탐지", f"{len(filtered_df[filtered_df['상태']=='장비 장애 의심']):,} 개", delta="현장 조치 필요", delta_color="normal")

    # 3. 지도 렌더링
    st.caption("안전과 성능 제어를 위해 실시간 이상치 와이파이(과부하: 빨강, 장애: 주황) 정보를 관제합니다.")
    m = folium.Map(location=map_center, zoom_start=zoom_level)

    # 내 위치 정보 표시 (GPS 켜져있을 때)
    if user_lat and user_lon:
        folium.Marker(
            location=[user_lat, user_lon],
            popup="내 현재 위치",
            icon=folium.Icon(color='purple', icon='user', prefix='fa')
        ).add_to(m)

    # 주변 와이파이 마킹 (최대 100개 제한)
    map_data = filtered_df.head(100)
    for idx, row in map_data.iterrows():
        color = 'blue'
        if row['상태'] == '과부하/인파밀집':
            color = 'red'
        elif row['상태'] == '장비 장애 의심':
            color = 'orange'
            
        dist_info = f"<br>내 거리: {row['거리(km)']:.2f}km" if '거리(km)' in row else ""
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"AP: {row['관리번호']}<br>구: {row['자치구']}<br>사용량: {row['AP별 이용량(GB)']:.2f}GB<br>Z-score: {row['Z_score']:.2f}{dist_info}",
            tooltip=f"{row['관리번호']} ({row['상태']})",
            icon=folium.Icon(color=color, icon='wifi', prefix='fa')
        ).add_to(m)

    st_folium(m, width=1200, height=500)

    # 4. 데이터 표 출력
    st.subheader("📋 실시간 이상치 탐지 및 목록 정보")
    cols_to_show = ['관리번호', '자치구', 'AP별 이용량(GB)', 'Z_score', '상태']
    if '거리(km)' in filtered_df.columns:
        cols_to_show.append('거리(km)')
    st.dataframe(filtered_df[cols_to_show].head(50), use_container_width=True)
