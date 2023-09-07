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

import to_upward
from tqdm.auto import tqdm
from streamlit_extras.switch_page_button import switch_page

from pages import shape
from pages import correlation
from pages import chatbot

if "page" not in st.session_state:
    st.session_state.page = "home"


DATA_PATH = "C:/Users/Jonghyeon/Desktop/파이널프로젝트/data/"
SEED = 42

# 데이터 불러오는 함수(캐싱)
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_csv(path):
    return pd.read_csv(path)

# 데이터 불러오기
# data = load_csv(f"{DATA_PATH}labeled_data_final2.csv")

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


data = load_csv(f"{DATA_PATH}labeled_data_final2.csv")


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




    st.write("이 중 기준으로 할 하나의 기업을 선택하세요.")

    selected_result = st.selectbox('기준 선택', st.session_state.recommended_stocks ,help= '선택한 기업을 기준으로 상관계수가 낮은 순서대로 다른 기업들이 자동 선택됩니다.')
    st.write('선택한 결과: ', selected_result)
    if selected_result is not None:
        #try:

            str_list = data.Code.astype(str).to_list()
            target_len = 6
            padded_str_list = to_upward.pad_str(str_list, target_len)
            data.Code = padded_str_list



            @st.cache_data(ttl=900)
            def load_stock(start, end, data, stocks):
                t= pd.DataFrame()
                for n in stocks:
                    t[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']
                return t
            tmp= load_stock(start, end, data, st.session_state.recommended_stocks)

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

                port_ret = []
                port_risk = []
                port_weights = []
                shape_ratio = []
                rf = 0.0325
                for i in range(30000):
                    weights = np.random.random(len(stocks))
                    weights /= np.sum(weights)
                    
                    returns = np.dot(weights, annual_ret)
                    risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
                    port_ret.append(returns)
                    port_risk.append(risk)
                    port_weights.append(weights)
                    shape_ratio.append(returns/risk)
                portfolio = {'Returns' : port_ret, 'Risk' : port_risk, 'Shape' : shape_ratio}
                for j, s in enumerate(stocks):
                    portfolio[s] = [weight[j] for weight in port_weights]
                df = pd.DataFrame(portfolio)
                max_shape = df.loc[df['Shape'] == df['Shape'].max()]
                min_risk = df.loc[df['Risk'] == df['Risk'].min()]
                tmp2 = df.groupby('Risk')[['Returns']].max().reset_index()

                best_ret = tmp2.loc[0,'Returns']
                for i in range(tmp2.shape[0]):
                    if tmp2.loc[i,'Returns']<best_ret:
                        tmp2.drop(index=i,inplace=True)
                    elif tmp2.loc[i, 'Returns'] >= best_ret:
                        best_ret = tmp2.loc[i,'Returns']
                import plotly.graph_objects as go

                def show_CAPM(df, tmp2, max_shape, min_risk, rf=0.035):
                    df.plot.scatter(x='Risk', y='Returns', c='Shape', cmap='viridis', edgecolors='k', figsize=(10,8), grid=True)
                    plt.plot(tmp2['Risk'], tmp2['Returns'], label='Efficient Frontier', linewidth=5,color='red')
                    plt.scatter(max_shape['Risk'], max_shape['Returns'], label='Max_Shape', marker='*',s=500)
                    plt.scatter(min_risk['Risk'], min_risk['Returns'], label='Min_risk', marker='*', s=500)
                    plt.plot([0, max_shape['Risk'].iloc[0], 0.5], [rf, max_shape['Returns'].iloc[0], (max_shape['Returns'].iloc[0] - rf) / max_shape['Risk'].iloc[0] * 0.5 + rf], label='New EF', linewidth=2,color='green')
                    plt.xlabel('Risk')
                    plt.ylabel('Expected Return')
                    plt.title('Efficient Frontier Graph')
                    plt.legend()
                    st.pyplot(plt)

                show_CAPM(df,tmp2,max_shape,min_risk,rf=0.035)
                st.write('max_shape')
                st.dataframe(max_shape)
                st.write('min_risk')
                st.dataframe(min_risk)
                rf=0.0325
                min_value= (f"{100*rf:.2f}")
                min_value= float(min_value)
                max_value= (f"{200:.2f}")
                max_value= float(max_value)
                max_return= (f"{100*max_shape['Returns'].iloc[0]:.2f}")
                max_return = float(max_return)
                st.session_state.exp_ret = st.slider("기대수익을 선택해주세요.", min_value, max_value, step=0.1, key="corr_1") /100
                st.text(f"위험기피: 기대수익 {min_value}% 이상 {max_return}% 미만입니다.\n중립: 기대수익 {max_return}% 입니다.\n위험선호: 기대수익 {max_return}% 초과 {max_value}% 이하입니다.")
                if float(st.session_state.exp_ret) >= min_value and float(st.session_state.exp_ret) < max_return:
                    st.write('당신은 안정형(위험기피) 유형입니다.')
                elif float(st.session_state.exp_ret) == max_return:
                    st.write('당신은 안정형(중립) 유형입니다.')
                elif float(st.session_state.exp_ret) > max_return and float(st.session_state.exp_ret) <= max_value:
                    st.write('당신은 안정형(위험선호) 유형입니다. 수익형 포트폴리오를 선택하시는 것을 권장합니다.')

                st.divider()

                if st.session_state.exp_ret is not None:

                    w = sympy.Symbol('w')

                    equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - st.session_state.exp_ret

                    solution = sympy.solve(equation, w)
                    solution = float(solution[0])
                    if solution < 0:

                        st.write(f"채권의 비중 : {-solution}")
                        st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

                    else:
                        st.write(f"채권의 비중 : {solution}")
                        st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")


                    if solution >= 0:

                        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", "<b>기대수익을 위한 포트폴리오"))


                        fig.add_trace(go.Pie(
                            values=list(max_shape.values[0][3:]),
                            labels=list(max_shape.columns[3:]),
                            domain=dict(x=[0, 0.5]),
                            name="기존 포트폴리오"),
                            row=1, col=1)

                        fig.add_trace(go.Pie(
                            values=list(max_shape.values[0][3:]* (1-float(solution)))+[float(solution)] ,
                            labels=list(max_shape.columns[3:]) + ['채권'],
                            domain=dict(x=[0.5, 1.0]),
                            name="기대수익 포트폴리오"),
                            row=1, col=2)

                        st.plotly_chart(fig)

                    else:
                        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", f"<b>투자금 비중</b><br><sup>자기자본의 {-solution*100:0.4}%만큼 차입</sup>"))


                        fig.add_trace(go.Pie(
                            values=list(max_shape.values[0][3:]),
                            labels=list(max_shape.columns[3:]),
                            domain=dict(x=[0, 0.5])),
                            row=1, col=1)

                        fig.add_trace(go.Pie(
                            values=[1/(1-solution),1-(1/(1-solution))] ,
                            labels=['자기자본','차입금'],
                            domain=dict(x=[0.5, 1.0])),
                            row=1, col=2)
                        st.plotly_chart(fig)
                    st.divider()

                    want_to_monte = st.button("몬테카를로 시뮬레이션 결과 보기")
                    st.text(f'몬테카를로 시뮬레이션은 불확실한 상황에서 수치적 예측을 수행하는 데 사용되는 통계적 방법입니다.\n이를 활용해 위의 포트폴리오로 투자했을 때 얼마의 수익을 얻을 수 있다고 가정할 수 있는지\n1000번의 시뮬레이션을 돌려 추후 5달간의 예측값을 상위 10%, 25%, 50%, 75%, 90%로 제시합니다.')
                    if want_to_monte:
                        sim_num=1000
                        balance = 1000000
                        stock_money= max_shape[max_shape.columns[3:]]*balance
                        balance_df= to_upward.monte_sim(sim_num,tmp,stocks,stock_money)
                        tmp3 = pd.DataFrame()
                        for i in [0.9,0.75,0.5,0.25,0.1]:
                            lst = []
                            idx = balance_df.T[balance_df.iloc[-1] >= balance_df.iloc[-1].quantile(i)][99].sort_values().index[0]
                            for k in range(19,100,20):
                                lst.append((balance_df.T.iloc[idx].iloc[k]-balance)/balance*100)
                            tmp3[f'{100-i*100}%'] = lst

                        tmp3.index=[f"{i}month" for i in range(1,6)]
                        st.write(tmp3)
                        st.write(px.line(tmp3))
                        st.success("메인 화면으로 돌아가려면 상단의 메인화면 버튼을 눌러주세요.")

                else:
                    st.warning('기대 수익을 선택해주세요.')


           
except Exception as e:
    pass
    
