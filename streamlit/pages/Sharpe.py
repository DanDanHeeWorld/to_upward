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
from pages import Sharpe
from pages import Correlation
from pages import Stock_Chatbot
import to_upward
from streamlit_extras.switch_page_button import switch_page
import io
import base64
import scipy.optimize as sco

if "page" not in st.session_state:
    st.session_state.page = "home"


DATA_PATH = "/"
SEED = 42

# 데이터 불러오는 함수(캐싱)
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_csv(path):
    return pd.read_csv(path)

# 데이터 불러오기
data = load_csv(f"{DATA_PATH}labeled_data_final2.csv")


# 마감일 및 시작일
end_03 = dt.datetime(2023,3,1).strftime("%Y%m%d")
start_03 = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
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

if 'recommended_stocks' not in st.session_state:
    st.session_state.recommended_stocks = []

warnings.filterwarnings('ignore')


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
    data = pd.read_csv(f"{DATA_PATH}labeled_data_final2.csv")
    st.markdown(f'''투자 유형: :red[**{st.session_state.type_of_user}**]''')
    st.markdown(f'''선택한 섹터 : :blue[**{st.session_state.selected_sectors}**]''')
    st.markdown(f'''추천 주식 : :green[**{st.session_state.recommended_stocks}**]''')

    if len(st.session_state.recommended_stocks) >1:
            str_list = data.Code.astype(str).to_list()
            target_len = 6
            padded_str_list = to_upward.pad_str(str_list, target_len)
            data.Code = padded_str_list



            tmp=to_upward.get_close(data,st.session_state.recommended_stocks,start,end)
            before_data = to_upward.get_close(data,st.session_state.recommended_stocks,start_03,end_03)
            now_data = to_upward.get_close(data,st.session_state.recommended_stocks,start,end)
            kospi200 = stock.get_index_ohlcv_by_date(start_03, end, "1028")['종가']
            daily_ret = tmp[st.session_state.recommended_stocks].pct_change()
            annual_ret = (1+daily_ret.mean())**tmp[st.session_state.recommended_stocks].shape[0]-1
            daily_cov = daily_ret.cov()
            annual_cov = daily_cov * tmp[st.session_state.recommended_stocks].shape[0]

            tmp2 = pd.DataFrame((annual_ret-0.02)/daily_ret.std()*np.sqrt(252),columns= ['Shape']).sort_values(by='Shape',ascending=False)

            if len(st.session_state.recommended_stocks) >=5:
                stocks = list(tmp2.iloc[0:5].index)
            elif len(st.session_state.recommended_stocks) <5:
                stocks= list(tmp2.iloc[0:].index)

            daily_ret = tmp[stocks].pct_change()
            annual_ret = (1+daily_ret.mean())**tmp[stocks].shape[0]-1
            daily_cov = daily_ret.cov()
            annual_cov = daily_cov * tmp[stocks].shape[0]

            
            if sum(annual_ret<0) == len(annual_ret<0):
                st.warning(f'연평균 수익률이 모두 음수인 업체이므로, 포트폴리오를 구성하기에 바람직하지 않습니다. 새롭게 sector를 선택해주세요.')

            else:
                col1, col2= st.columns(2)
                with col1:
                    tmp2
                with col2:
                    st.write('상위 기업 자동 선택:', stocks)
                st.divider()

                col3, col4= st.columns(2)
                max_shape,min_risk,tmp2,df=to_upward.get_portfolio(stocks,annual_ret,annual_cov)
                with col3:
                    to_upward.show_CAPM(df, tmp2, max_shape, min_risk, rf=0.035)
                with col4:
                    st.write('최대 샤프 비율')
                    st.dataframe(max_shape)
                    st.write('최소 리스크 비율')
                    st.dataframe(min_risk)

                min_value= (f"{4:.2f}")
                min_value= float(min_value)
                max_value= (f"{200:.2f}")
                max_value= float(max_value)
                max_return= (f"{100*max_shape['Returns'].iloc[0]:.2f}")
                max_return = float(max_return)
                
                exp_ret = st.slider("기대수익을 선택해주세요.", min_value, max_value, step=0.1,key="slider_sharpe") /100
                col5, col6= st.columns(2)
                with col5:
                    st.markdown(f'''위험 기피: :green[**기대수익 {min_value}% 이상 {max_return}% 미만입니다.**]''')
                    st.markdown(f'''중립: :orange[**기대수익 {max_return}% 입니다.**]''')
                    st.markdown(f'''위험 선호: :red[**기대수익 {max_return}% 초과 {max_value}% 이하입니다.**]''')
                with col6:
                    if exp_ret*100 >= min_value and exp_ret*100 < max_return:
                        st.image('위험기피.png', caption='당신은 수익형(위험기피형)입니다.', use_column_width=True)
                    elif exp_ret*100 == max_return:
                        st.image('중립.png', caption='당신은 수익형(중립형)입니다.', use_column_width=True)
                    elif exp_ret*100 > max_return and exp_ret*100 <= max_value:
                        st.image('위험선호.png', caption='당신은 수익형(위험선호형)입니다', use_column_width=True)

                st.divider()

                fig, solution=to_upward.show_portfolio(max_shape,exp_ret)
                to_upward.show_portfolio2(fig, solution,max_shape)
                st.divider()
                if exp_ret is not None:
                    tab1, tab2= st.tabs(['미래','과거'])

                    with tab1:
                        st.text(f'몬테카를로 시뮬레이션은 불확실한 상황에서 수치적 예측을 수행하는 데 사용되는 통계적 방법입니다.\n이를 활용해 위의 포트폴리오로 투자했을 때 얼마의 수익을 얻을 수 있다고 가정할 수 있는지\n1000번의 시뮬레이션을 돌려 추후 5달간의 예측값을 상위 10%, 25%, 50%, 75%, 90%로 제시합니다.')
                        sim_num=1000
                        balance = 1000000
                        stock_money= max_shape[max_shape.columns[3:]]*balance
                        balance_df= to_upward.monte_sim(sim_num,tmp,stocks,stock_money)
                        tmp3=to_upward.get_simret(balance_df,balance,before_data,stocks,max_shape,solution,None,None,rf=0.0325)
                        col7, col8= st.columns(2)
                        with col7:
                            st.write(tmp3)
                        with col8:                 
                            tmp4 = {'호황': [tmp3['호황'][4]*balance],
                                    '상승': [tmp3['상승'][4]*balance],
                                    '평년': [tmp3['평년'][4]*balance],
                                    '하락': [tmp3['하락'][4]*balance],
                                    '불황': [tmp3['불황'][4]*balance]}

                            df4 = pd.DataFrame(tmp4)

                            df4.index = ['예상']
                            st.table(df4)

                        st.write(px.line(tmp3))
                    
                    with tab2:
                        st.text(f'몬테카를로 시뮬레이션은 불확실한 상황에서 수치적 예측을 수행하는 데 사용되는 통계적 방법입니다.\n이를 활용해 올해 3월로 돌아가 위의 포트폴리오로 투자했을 때 얼마의 수익을 얻었다고 할 수 있는지\n1000번의 시뮬레이션을 돌려 8월까지의 예측값을 상위 10%, 25%, 50%, 75%, 90%로 제시해 실제 값과 비교하여 제시합니다.')
                        daily_ret = before_data[st.session_state.recommended_stocks].pct_change()
                        annual_ret = (1+daily_ret.mean())**before_data[st.session_state.recommended_stocks].shape[0]-1
                        daily_cov = daily_ret.cov()
                        annual_cov = daily_cov * before_data[st.session_state.recommended_stocks].shape[0]
                        tmp2 = pd.DataFrame((annual_ret-0.02)/daily_ret.std()*np.sqrt(252),columns= ['Shape']).sort_values(by='Shape',ascending=False)
                        if len(st.session_state.recommended_stocks) >=5:
                            stocks = list(tmp2.iloc[0:5].index)
                        elif len(st.session_state.recommended_stocks) <5:
                            stocks= list(tmp2.iloc[0:].index)
                        daily_ret = before_data[stocks].pct_change()
                        annual_ret = (1+daily_ret.mean())**before_data[stocks].shape[0]-1
                        daily_cov = daily_ret.cov()
                        annual_cov = daily_cov * before_data[stocks].shape[0]
                        rf = 0.0325
                        max_shape,min_risk,tmp2,df = to_upward.get_portfolio(stocks,annual_ret,annual_cov)
                        sim_num=1000
                        balance = 1000000
                        stock_money= max_shape[max_shape.columns[3:]]*balance
                        balance_df= to_upward.monte_sim(sim_num,tmp,stocks,stock_money)
                        tmp3=to_upward.get_simret(balance_df,balance,before_data,stocks,max_shape,solution,now_data,kospi200,rf=0.0325)
                        col9, col10= st.columns(2)
                        with col9:
                            st.write(tmp3)
                        with col10:
                            tmp4 = {'호황': [tmp3['호황'][4]*balance],
                                    '상승': [tmp3['상승'][4]*balance],
                                    '평년': [tmp3['평년'][4]*balance],
                                    '하락': [tmp3['하락'][4]*balance],
                                    '불황': [tmp3['불황'][4]*balance]}

                            df4 = pd.DataFrame(tmp4)

                            df4.index = ['결과']
                            st.table(df4)

                        st.write(px.line(tmp3))
    else:
        st.warning("추천 주식의 개수가 1개이므로, 포트폴리오를 구성하기에 바람직하지 않습니다. 새롭게 sector를 선택해주세요.")
        
except Exception as e:
    pass




    
        




