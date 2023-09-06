import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from pykrx import stock
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy
from pages import correlation
from pages import shape
from pages import chatbot2
import to_upward
from streamlit_extras.switch_page_button import switch_page
if "page" not in st.session_state:
    st.session_state.page = "home"


DATA_PATH = "./"
SEED = 42

# 데이터 불러오는 함수(캐싱)
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_csv(path):
    return pd.read_csv(path)

# 데이터 불러오기
data = load_csv(f"{DATA_PATH}labeled_data_final2.csv")

# 오류 방지를 위한 패딩 함수
def pad_str(str_list, target_len):

  padded_str_list = []
  for str in str_list:
    if len(str) < target_len:
      padded_str = "0" * (target_len - len(str)) + str
    else:
      padded_str = str
    padded_str_list.append(padded_str)
  return padded_str_list
str_list = data.Code.astype(str).to_list()
target_len = 6
padded_str_list = pad_str(str_list, target_len)

data.Code = padded_str_list

# 마감일 및 시작일
end = dt.datetime.today().date().strftime("%Y%m%d")
start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")

# 종가를 가져올 주식 목록
all_stocks = data['Name'] # 전체 선택

# pykrx에서 종가 데이터를 가져오는 함수
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_stock(start, end, data, all_stocks):
    t = pd.DataFrame()
    for n in all_stocks:
        t[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']
    return t

tmp = load_stock(start, end, data, all_stocks)





def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)




PAGES = {

    "Shape Portfolio": shape,
    "Correlation" : correlation,
    "chatbot" : chatbot2

}


if 'type_of_user' not in st.session_state:
    st.session_state.type_of_user = None

if 'selected_sectors' not in st.session_state:
    st.session_state.selected_sectors = []

if 'exp_ret' not in st.session_state:
    st.session_state.exp_ret = 5.0

if 'recommended_stocks' not in st.session_state:
    st.session_state.recommended_stocks = []


warnings.filterwarnings('ignore')



# Survey Part
st.title("우상향, 나의 투자 포트폴리오")
st.title("설문조사")




sectors = ["dividend", "growth", "value", "performance", "business", "finance", "volitality"]

# 필터링
def filter_by_grade(data, sector):
    if sector in ['finance']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C') | (data[sector] == 'D') | (data[sector] == 'E')
    elif sector in ['volitality']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C')
    elif sector in ['business']:
        return (data[sector] == 'A') | (data[sector] == 'B')
    elif sector in ['dividend']:
        return (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C')
    elif sector in ['value']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C')
    elif sector in ['growth']:
        return (data[sector] == 'A') | (data[sector] == 'A-') | (data[sector] == 'B') | (data[sector] == 'B-')
    elif sector in ['performance']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C')



if 'selected_sectors' not in st.session_state:
    st.session_state.selected_sectors = st.multiselect("중요하게 여기는 가치 2가지~3가지를 선택하세요.", options=sectors,max_selections=3 )
else:
    st.session_state.selected_sectors = st.multiselect("중요하게 여기는 가치 2가지~3가지를 선택하세요.", options=sectors, max_selections=3,default=st.session_state.selected_sectors)


if len(st.session_state.selected_sectors) ==2:
    st.success(f"선택하신 섹터는 {', '.join(st.session_state.selected_sectors)} 입니다.")


    stocks = []

    # 섹터 선택 확인
    if st.session_state.selected_sectors:
        conditions = [filter_by_grade(data, sector) for sector in st.session_state.selected_sectors]
        final_condition = np.logical_and.reduce(conditions)
        stocks = data[final_condition]["Name"].to_list()
        st.write('추천 종목:', stocks)
        # 추천 종목을 표시
        # 추천 종목 계산 후
        st.session_state.recommended_stocks = stocks
        st.write('당신의 투자 성향을 선택해주세요.')
        st.markdown(
            """
            <style>
            .stButton > button {
                background-color: #F8FFD3;
                width: 100%; /
                display: inline-block;
                margin: 0; /
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        def page1():
            want_to_corr = st.button("안정형")
            if want_to_corr:
                st.session_state.type_of_user = "안정형"
                switch_page("correlation")


        def page2():

            want_to_shape = st.button("수익형")
            if want_to_shape:
                st.session_state.type_of_user = "수익형"
                switch_page("shape")


        col1, col2 = st.columns(2)
        with col1:
            page1()
        with col2:
            page2()

elif len(st.session_state.selected_sectors) == 3:
    st.success(f"선택하신 섹터는 {', '.join(st.session_state.selected_sectors)} 입니다.")


    stocks = []

    # 섹터 선택 확인
    if st.session_state.selected_sectors:
        conditions = [filter_by_grade(data, sector) for sector in st.session_state.selected_sectors]
        final_condition = np.logical_and.reduce(conditions)
        stocks = data[final_condition]["Name"].to_list()
        st.write('추천 종목:', stocks)
        # 추천 종목을 표시
        # 추천 종목 계산 후
        st.session_state.recommended_stocks = stocks


        st.write('당신의 투자 성향을 선택해주세요.')


        st.markdown(
            """
            <style>
            .stButton > button {
                background-color: #F8FFD3;
                width: 100%; /
                display: inline-block;
                margin: 0; /
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        def page1():
            want_to_corr = st.button("안정형")
            if want_to_corr:
                st.session_state.type_of_user = "안정형"
                switch_page("correlation")


        def page2():

            want_to_shape = st.button("수익형")
            if want_to_shape:
                st.session_state.type_of_user = "수익형"
                switch_page("shape")


        col1, col2 = st.columns(2)
        with col1:
            page1()
        with col2:
            page2()



        
else:
    st.write('섹터를 2개 이상 선택해주세요.')



def init():
    if 'show_description' not in st.session_state:
        st.session_state.show_description = False

init()

# 위젯으로 "주식 섹터에 대한 설명" 보기/숨기기
if st.sidebar.checkbox('주식 섹터에 대한 설명 보기', help='클릭 시 설명을 확인할 수 있습니다.'):
    st.session_state.show_description = True
else:
    st.session_state.show_description = False

# "주식 섹터에 대한 설명"이 True면 내용을 표시
if st.session_state.show_description:
    st.write("### 주식 섹터에 대한 설명")

    st.write("#### 배당 (Dividend)")
    if st.button('더 보기', key='dividend'):
        st.write("""
        **설명**: 주식회사가 주주에게 이익을 나눠주는 것을 의미합니다. 배당률이 높은 주식은 안정적인 수익을 기대할 수 있습니다.\n
        **예시**: 대한민국에서는 삼성전자, SK텔레콤 등이 배당률이 높은 주식으로 알려져 있습니다.
        """)

    st.write("#### 성장 (Growth)")
    if st.button('더 보기', key='growth'):
        st.write("""
        **설명**: 회사의 매출이나 이익이 지속적으로 증가하는지를 살펴보는 지표입니다. 성장률이 높은 주식은 높은 수익률을 기대할 수 있습니다.\n
        **예시**: NAVER, 카카오 등은 지속적인 성장을 보이는 주식입니다.
        """)    

    st.write("#### 가치 (Value)")
    if st.button('더 보기', key='value'):
        st.write("""
        **설명**: 주식의 현재 가격이 그 실제 가치에 비해 얼마나 저렴한지를 평가하는 지표입니다. P/E 비율, P/B 비율 등을 통해 측정합니다.\n
        **예시**: POSCO, 현대차 등은 가치 투자의 대상으로 여겨지는 경우가 많습니다.
        """)    

    st.write("#### 경영 (Business)")
    if st.button('더 보기', key='business'):
        st.write("""
        **설명**: 회사의 경영 성과를 평가하는 지표입니다. 영업이익률, 순이익률 등을 통해 회사의 경영 상태를 파악합니다.\n
        **예시**: 삼성바이오로직스, 셀트리온 등은 높은 영업이익률을 가진 회사입니다.
        """)    

    st.write("#### 재무 (Finance)")
    if st.button('더 보기', key='finance'):
        st.write("""
        **설명**: 회사의 재무 상태를 평가하는 지표입니다. 부채비율, 유동비율 등을 통해 회사의 재무 안정성을 측정합니다.\n
        **예시**: 삼성SDI, LG화학 등은 재무상태가 안정적인 회사로 알려져 있습니다.
        """)    

    st.write("#### 실적 (Performance)")
    if st.button('더 보기', key='performance'):
        st.write("""
        **설명**: 회사의 경영 실적을 평가하는 지표입니다. 매출, 이익, EBITDA 등을 통해 회사의 경영 성과를 측정합니다.\n
        **예시**: 삼성전자, SK하이닉스 등은 높은 매출과 이익을 보이는 회사입니다.
        """)    

    st.write("#### 변동성 (Volatility)")
    if st.button('더 보기', key='volatility'):
        st.write("""
        **설명**: 주식 가격의 변동 폭을 의미합니다. 변동성이 높은 주식은 높은 수익, 높은 리스크를 가질 수 있습니다.\n
        **예시**: 바이오 관련 주식이나 새로운 기술을 개발한 스타트업 주식 등은 변동성이 높은 경우가 많습니다.
        """)    

