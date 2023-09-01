
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
from pages import shape,corr
import Home



def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

DATA_PATH = "./"
SEED = 42


PAGES = {
    "Home": Home,
    "Shape Portfolio": shape,
    "Correlation" : corr
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()



if 'user_type' not in st.session_state:
    st.session_state.user_type = None

if 'selected_sectors' not in st.session_state:
    st.session_state.selected_sectors = []

if 'exp_ret' not in st.session_state:
    st.session_state.exp_ret = 5.0

if 'recommended_stocks' not in st.session_state:
    st.session_state.recommended_stocks = []


warnings.filterwarnings('ignore')


data = pd.read_csv(f"{DATA_PATH}labeled_data_final.csv")

value_df = pd.read_csv(f"{DATA_PATH}value_df2.csv")


# Survey Part
st.title("우상향, 나의 투자 포트폴리오")
st.title("설문조사")

type_of_user = st.radio("당신의 투자 성향은..?", ["수익형", "안정형"])
if type_of_user == "안정형":
    st.write(
        """
        - 주식투자에서 가장 중요한 것은 원금을 지키는 것입니다. 
        - 안정적인 수익을 추구하며, 큰 리스크를 회피합니다. 
        - 주로 대형주, 배당주 위주로 투자하는 경향이 있습니다.
        """
    )
else:
    st.write(
        """
        - 주식투자에서 큰 수익을 추구합니다. 
        - 그에 따른 큰 리스크를 감수할 준비가 되어 있습니다. 
        - 주로 소형주, 성장주 위주로 투자하는 경향이 있습니다.
        """
    )
st.session_state.type_of_user = type_of_user


sectors = ["dividend", "growth", "value", "performance", "business", "finance", "performance", "volitality"]


if 'selected_sectors' not in st.session_state:
    st.session_state.selected_sectors = st.multiselect("중요하게 여기는 가치 3가지를 선택하세요.", options=sectors)
else:
    st.session_state.selected_sectors = st.multiselect("중요하게 여기는 가치 3가지를 선택하세요.", options=sectors, default=st.session_state.selected_sectors)



if len(st.session_state.selected_sectors) > 3:
    st.warning("3개 이상의 섹터를 선택해주세요.")
elif len(st.session_state.selected_sectors) == 3:
    st.success(f"선택하신 섹터는 {', '.join(st.session_state.selected_sectors)} 입니다.")


# 필터링
def filter_by_grade(data, sector):
    if sector in ['finance']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C') | (data[sector] == 'D')
    elif sector in ['value','volitality']:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B') | (data[sector] == 'C')
    else:
        return (data[sector] == 'A+') | (data[sector] == 'A') | (data[sector] == 'B')


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
    
else:
    st.write('섹터를 선택해주세요.')


if 'exp_ret' not in st.session_state:
    st.session_state.exp_ret = st.slider('기대수익률 설정', 0.0, 20.0, 5.0, key="exp_ret_slider")
else:
    st.session_state.exp_ret = st.slider('기대수익률 설정', 0.0, 20.0, st.session_state.exp_ret, key="exp_ret_slider")




def app():
    st.title("Home Page")