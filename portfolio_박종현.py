# 패키지 설치(깃허브 원격저장소에 구성해서 배포시 requirements.txt로 에러 방지)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pykrx import stock
import plotly.express as px
import matplotlib.pyplot as plt
import koreanize_matplotlib
import datetime as dt
import sympy
from plotly.subplots import make_subplots
import random
import os
import warnings
warnings.filterwarnings('ignore')

# 시드 및 경로 설정
def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

DATA_PATH = "C:/Users/Jonghyeon/Desktop/파이널프로젝트/data/"
SEED = 42

# 데이터 불러오기
# data = pd.read_csv(f"{DATA_PATH}preprocessing_data5.csv")
data = pd.read_csv(f"{DATA_PATH}labeled_data_final.csv")


# 종료일 및 시작일
end = dt.datetime.today().date().strftime("%Y%m%d")
start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")


# 종가 데이터 불러오기
tmp = pd.read_csv(f"{DATA_PATH}test_stock2.csv", index_col=0, parse_dates=True) # 인덱스를 날짜로 불러오기 

# sector 기준
dividend = ['dividendYield','dividendRate','5년평균dividendYield']
growth = ['revenueGrowth','earningsGrowth','earningsQuarterlyGrowth','revenueQuarterlyGrowth','heldPercentInsiders']
value = ['priceToBook','enterpriseValue','enterpriseToRevenue','enterpriseToEbitda','trailingEps','priceToSalesTrailing12Months','trailingPE']
business = ['returnOnAssets','returnOnEquity','grossMargins','operatingMargins','profitMargins']
finance = ['debtToEquity','operatingCashflow','freeCashflow','totalCashPerShare','currentRatio','quickRatio','totalCash','totalDebt','BPS']
performance = ['totalRevenue','grossProfits','revenuePerShare','ebitdaMargins','EBITDAPS']
volitality = ['marketCap','currentPrice','fiftyDayAverage','twoHundredDayAverage','52WeekChange','ytdReturn','fiveYearAverageReturn','beta']




# 메인화면에 표시할 부분 (st.)
st.title('투자성향에 맞는 포트폴리오 추천')
st.divider()

# <<<<< 사이드바 >>>>>
st.sidebar.title('포트폴리오 선택')

# 사이드바에서 투자 성향 선택
selected_item = st.sidebar.radio("(1) 당신의 투자 성향을 선택하세요:", ('안정형', '중립형', '수익형'))

# 투자 성향에 따른 포트폴리오 제공
# 1.원하는 기업 하나 선택(소비자가 선택하지 않으면 어려운 숙제)
# 2.그대로 활용 vs 기대수익에 맞게 구성
# 3.고객이 잘못된 정보를 입력했을 때 오류메세지 처리하기


# 라벨링 기준에 따른 분류
labelling = {
    'dividend': ['A', 'B', 'C'],
    'growth': ['A', 'A-', 'B', 'B-'],
    'value': ['A', 'B', 'C'],
    'business': ['A', 'B', 'C'],
    'finance': ['A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'performance': ['A+', 'A', 'B'],
    'volitality': ['A+', 'A', 'B', 'C']
}

if selected_item == "안정형":
    multi_select = st.sidebar.multiselect('Sector 3가지를 선택하세요', list(labelling.keys()), max_selections=3, placeholder='sector를 선택하세요.') 

    # multi_select 조건문
    st.write('당신의 선택:', multi_select)

    conditions = []
    for idx, selected in enumerate(multi_select):
        condition = data[selected].isin(labelling[selected])
        conditions.append(condition)

    recommendation = data[np.logical_and.reduce(conditions)] # 여러 조건을 동시에 만족하는 경우
    selected_stock = recommendation["Name"].to_list()
    tmp_stock = tmp[selected_stock]

    st.write('추천 종목:', selected_stock)



    selected_stock = st.sidebar.selectbox("목록 중 원하는 기업을 선택하세요.", selected_stock)
    
    # selected_stock = st.sidebar.text_input('원하는 기업 선택', '삼성전자') # 원하는 기업을 먼저 선택할 수 있게??
    # 선택한 기업 + 상관계수 상위 4개 기업
    stocks = list(tmp_stock.corr()[[selected_stock]].sort_values(by=selected_stock)[:4].index)+[selected_stock] # 상위 4개 선정
    st.write('선택된 기업과 상위 4개의 기업 포트폴리오')
    st.dataframe(stocks)

    daily_ret = tmp[stocks].pct_change()
    annual_ret = daily_ret.mean() * tmp[stocks].shape[0]

    daily_cov = daily_ret.cov()
    annual_cov = daily_cov * tmp[stocks].shape[0]



    # 포트폴리오 시각화
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

    # best_ret = tmp2.loc[0,'Returns']
    # for i in range(tmp2.shape[0]):
    #     if tmp2.loc[i,'Returns']<best_ret:
    #         tmp2.drop(index=i,inplace=True)
    # else:
    #     best_ret = tmp2.loc[i,'Returns']

    best_ret = tmp2.loc[0,'Returns']
    for i in range(tmp2.shape[0]):
        if tmp2.loc[i,'Returns'] < best_ret:
            tmp2.drop(index=i, inplace=True)
        # else로 할 경우 i 값은 for 루프의 마지막 값을 가지게 되며, 그 값이 tmp2 데이터프레임에 존재하지 않을 경우 KeyError가 발생
        elif tmp2.loc[i,'Returns'] >= best_ret: 
            best_ret = tmp2.loc[i,'Returns']

    fig = go.Figure()


    fig.add_trace(go.Scatter(x=df['Risk'], y=df['Returns'], mode='markers',name='Portfolio',marker=dict(
            size=5,    # 점 크기
            color=df['Shape'],
            colorscale = 'earth',
            showscale=True,  # colorscales 보여줌
            colorbar={"title": "Shape"},
            line_width=1, # 마커 라인 두께 설정
    )))


    fig.add_trace(go.Scatter(x=tmp2['Risk'], y=tmp2['Returns'],name='Efficient Frontier',line_width=5,mode='lines'))

    fig.add_trace(go.Scatter(x=max_shape['Risk'],y=max_shape['Returns'], mode='markers',name='Max_Shape',marker=dict(size =20,symbol='star')))
    fig.add_trace(go.Scatter(x=min_risk['Risk'],y=min_risk['Returns'], mode='markers',name='Min_risk',marker=dict(size =20,symbol='star')))

    fig.add_trace(go.Scatter(x=[0,max_shape['Risk'].iloc[0],0.5], y=[rf,max_shape['Returns'].iloc[0],(max_shape['Returns'].iloc[0] - rf)/max_shape['Risk'].iloc[0]*0.5+rf],name='New EF',line_width=5,mode='lines'))

    fig.update_layout(title='Efficient Frontier Graph',
                    xaxis_title='Risk',
                    yaxis_title='Expected Return')

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.05
    ))

    # 그래프 출력
    st.plotly_chart(fig) # fig.show()대신 이 코드로 출력

    # max_shape, min_risk 데이터프레임으로 보여주기
    st.write('max_shape')
    st.dataframe(max_shape)
    st.write('min_risk')
    st.dataframe(min_risk)




    # 원하는 기대 수익은 얼마인가?
    exp_ret = st.sidebar.slider('원하는 기대 수익', min_value=0.0, max_value=15.0, step=0.1)

    w = sympy.Symbol('w')

    equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret

    solution = sympy.solve(equation, w)
    solution = float(solution[0])

    st.write(f"채권의 비중 : {solution}")
    st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")


    # 수평선 표시
    st.divider()

    if solution >= 0:
        
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", "<b>기대수익을 위한 포트폴리오"))


        fig.add_trace(go.Pie(
            values=list(max_shape.values[0][-3:]),
            labels=list(max_shape.columns[-3:]),
            domain=dict(x=[0, 0.5]),
            name="GHG Emissions"),
            row=1, col=1)

        fig.add_trace(go.Pie(
            values=list(max_shape.values[0][-3:]* (1-float(solution)))+[float(solution)] ,
            labels=list(max_shape.columns[-3:]) + ['채권'],
            domain=dict(x=[0.5, 1.0]),
            name="CO2 Emissions"),
            row=1, col=2)

        # 그래프 출력
        st.plotly_chart(fig)

    else:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", f"<b>투자금 비중</b><br><sup>자기자본의 {-solution*100:0.4}%만큼 차입</sup>"))


        fig.add_trace(go.Pie(
            values=list(max_shape.values[0][-3:]),
            labels=list(max_shape.columns[-3:]),
            domain=dict(x=[0, 0.5])),
            row=1, col=1)

        fig.add_trace(go.Pie(
            values=[1/(1-solution),1-(1/(1-solution))] ,
            labels=['자기자본','차입금'],
            domain=dict(x=[0.5, 1.0])),
            row=1, col=2)

        # 그래프 출력
        st.plotly_chart(fig)




elif selected_item == "중립형":
    st.write("중립형은 추후 추가될 예정입니다.")



elif selected_item == "수익형":
    selected_num = st.sidebar.number_input('원하는 기대수익 비율')
    st.write("수익형은 상위 shape지수의 기업 5개를 선정")

# 사이드바 구분선
st.sidebar.divider()

# 정보 관련
selected_info = st.sidebar.radio("(2) 투자 관련 정보를 확인하세요:", ('성향테스트', '주식정보'))

# if selected_info == "성향테스트":
#     st.subheader("성향테스트")

    
#     # 각 선택지에 대한 점수
#     score_dict = {
#         "닭": 3,
#         "달걀": 1,
#         "둘 다 아니다": 2,
#         "개미": 1,
#         "베짱이": 3,
#         "둘 다 옳다": 2,
#         "흥부": 3,
#         "놀부": 1,
#         "도깨비": 2,
#     }

#     # 투자 성향 심리테스트(MBTI로 안정형인지 수익형인지 중립형인지 istj가 선호하는 것 등 생각하기)
#     question1 = st.selectbox("1.당신이 생각했을 때 '닭'이 먼저인가요, 아니면 '달걀'이 먼저인가요?", ["닭", "달걀", "둘 다 아니다"])
#     # 닭(이미 성장이 끝나서 가장 비싼 주식)=수익형, 달걀(알부터 성장해 나아가는 주식)=안정형, 둘 다 아니다=중립형
#     question2 = st.selectbox("2.당신은 '개미와 베짱이'에서 누구의 행동이 옳다고 생각하시나요?", ["개미", "베짱이", "둘 다 옳다"])
#     # 개미=안정형, 베짱이=수익형, 둘 다 옳다=중립형
#     question3 = st.selectbox("3.당신은 '흥부와 놀부'에서 누가 악역이라고 생각하시나요?", ("흥부", "놀부", "도깨비"))
#     # 흥부(악역)=수익형, 놀부(악역)=안정형, 도깨비=중립형
    
#     if st.button("제출"):
#         total_score = score_dict[question1] + score_dict[question2] + score_dict[question3]
        
#         if total_score <= 4:
#             selected_item = "안정형"
#             st.write(f"당신의 투자 성향은 '{selected_item}'입니다.")
#         elif 4 < total_score < 7:
#             selected_item = "중립형"
#             st.write(f"당신의 투자 성향은 '{selected_item}'입니다.")
#         elif total_score >= 7:
#             selected_item = "수익형"
#             st.write(f"당신의 투자 성향은 '{selected_item}'입니다.")


# LLM 모델도 여러개 테스트 # Few shot learning이므로 파인튜닝 과정 필요 없음
if selected_info == "주식정보":
    st.subheader("주식정보")



# 버튼으로 체크하는 방식
checkbox_btn = st.sidebar.checkbox("주식 정보 챗봇")

if checkbox_btn:
    st.write("관련 정보 Few shot learning로 보여주기")