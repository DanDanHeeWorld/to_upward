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
from tqdm.auto import tqdm
from streamlit_extras.switch_page_button import switch_page
import io
import base64

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

# 마감일 및 시작일
end = dt.datetime.today().date().strftime("%Y%m%d")
start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if 'type_of_user' not in st.session_state:
    st.session_state.type_of_user = None

if 'selected_sectors' not in st.session_state:
    st.session_state.selected_sectors = []

if 'exp_ret' not in st.session_state:
    st.session_state.exp_ret = 5.0

if 'recommended_stocks' not in st.session_state:
    st.session_state.recommended_stocks = []

warnings.filterwarnings('ignore')


data = pd.read_csv(f"{DATA_PATH}labeled_data_final2.csv")


try:

    st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #F8FFD3;
        display: inline-block;
        margin: 0; /
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    def page3():
        want_to_home = st.button("메인화면")
        if want_to_home:
            switch_page("Home")
    page3()


    st.write(f"type_of_user: {st.session_state.type_of_user}")
    st.write(f"선택한 섹터 : {st.session_state.selected_sectors}")
    st.write(f"추천 주식 : {st.session_state.recommended_stocks}")


    if len(st.session_state.recommended_stocks) >1:

        st.write("이 중 기준으로 할 하나의 기업을 선택하세요.")

        selected_result = st.selectbox('기준 선택', st.session_state.recommended_stocks ,help= '선택한 기업을 기준으로 상관계수가 낮은 순서대로 다른 기업들이 자동 선택됩니다.')
        st.write('선택한 결과: ', selected_result)
        if selected_result is not None:

                str_list = data.Code.astype(str).to_list()
                target_len = 6
                padded_str_list = to_upward.pad_str(str_list, target_len)
                data.Code = padded_str_list

                tmp=to_upward.get_close(data,st.session_state.recommended_stocks,start,end)
                selected_result_sorted=tmp.corr()[[f'{selected_result}']].sort_values(by=f'{selected_result}')
                mask = selected_result_sorted[f'{selected_result}']<1
                mask_sorted=tmp.corr()[[f'{selected_result}']][mask].sort_values(by=f'{selected_result}')
                mask_sorted

                if len(mask_sorted) >=5:
                    stocks = list(mask_sorted.index)[0:4]+[f"{selected_result}"]
                elif len(mask_sorted) <5:
                    stocks= list(mask_sorted.index)[0:]+[f"{selected_result}"]


                st.write('상관계수 상위 기업과 선택한 기준:', stocks)
                st.divider()

                daily_ret = tmp[stocks].pct_change()
                annual_ret = (1+daily_ret.mean())**tmp[stocks].shape[0]-1
                daily_cov = daily_ret.cov()
                annual_cov = daily_cov * tmp[stocks].shape[0]

                if sum(annual_ret<0) == len(annual_ret<0):
                        st.warning(f'연평균 수익률이 모두 음수인 업체이므로, 포트폴리오를 구성하기에 바람직하지 않습니다. 새롭게 sector를 선택해주세요.')
                    
                else:
                    max_shape,min_risk,tmp2,df=to_upward.get_portfolio(stocks,annual_ret,annual_cov)
                    to_upward.show_CAPM(df, tmp2, max_shape, min_risk, rf=0.035)
                    st.write('max_shape')
                    st.dataframe(max_shape)
                    st.write('min_risk')
                    st.dataframe(min_risk)

                    min_value= (f"{4:.2f}")
                    min_value= float(min_value)
                    max_value= (f"{200:.2f}")
                    max_value= float(max_value)
                    max_return= (f"{100*max_shape['Returns'].iloc[0]:.2f}")
                    max_return = float(max_return)
                    st.session_state.exp_ret = st.slider("기대수익을 선택해주세요.", min_value, max_value, step=0.1) /100
                    st.text(f"위험기피: 기대수익 {min_value}% 이상 {max_return}% 미만입니다.\n중립: 기대수익 {max_return}% 입니다.\n위험선호: 기대수익 {max_return}% 초과 {max_value}% 이하입니다.")
                    st.divider()

                    if st.session_state.exp_ret is not None:
                        to_upward.show_portfolio(max_shape,st.session_state.exp_ret)
                        st.divider()

                        want_to_monte = st.button("몬테카를로 시뮬레이션 결과 보기")
                        st.text(f'몬테카를로 시뮬레이션은 불확실한 상황에서 수치적 예측을 수행하는 데 사용되는 통계적 방법입니다.\n이를 활용해 위의 포트폴리오로 투자했을 때 얼마의 수익을 얻을 수 있다고 가정할 수 있는지\n1000번의 시뮬레이션을 돌려 추후 5달간의 예측값을 상위 10%, 25%, 50%, 75%, 90%로 제시합니다.')
                        if want_to_monte:
                            sim_num=1000
                            balance = 1000000
                            stock_money= max_shape[max_shape.columns[3:]]*balance
                            balance_df= to_upward.monte_sim(sim_num,tmp,stocks,stock_money)
                            to_upward.get_simret(balance_df,balance)
                            st.write('위 그래프를 다운로드하려면, 그래프 우측 상단의 Download plot as a png 버튼을 클릭하세요.')
                            st.success("메인 화면으로 돌아가려면 상단의 메인화면 버튼을 눌러주세요.")

                    else:
                        st.warning('기대 수익을 선택해주세요.')
    else:
        st.warning("추천 주식의 개수가 1개이므로, 포트폴리오를 구성하기에 바람직하지 않습니다. 새롭게 sector를 선택해주세요.")

except Exception as e:
    pass
    











