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
import time
# import warnings
# warnings.filterwarnings('ignore')


# 시드 및 경로 설정
def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

DATA_PATH = "C:/Users/Jonghyeon/Desktop/파이널프로젝트/data/"
SEED = 42

# 데이터 불러오는 함수(캐싱)
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_csv(path):
    return pd.read_csv(path)

# 데이터 불러오기
data = load_csv(f"{DATA_PATH}labeled_data_final.csv")

# 종가 데이터 불러오기
# tmp = pd.read_csv(f"{DATA_PATH}test_stock2.csv", index_col=0, parse_dates=True) # 인덱스를 날짜로 불러오기 

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
stocks = data['Name'] # 전체 선택

# pykrx에서 종가 데이터를 가져오는 함수
# ttl(수명) 매개변수: 해당 시간이 다 되어 함수를 다시 호출하면 앱은 이전에 캐시된 값을 모두 삭제하고 함수가 다시 실행
@st.cache_data(ttl=900)  # 캐싱 데코레이터
def load_stock(start, end, data, stocks):
    t = pd.DataFrame()
    for n in stocks:
        t[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']
    return t

tmp = load_stock(start, end, data, stocks)



# sector 기준
dividend = ['dividendYield','dividendRate','5년평균dividendYield']
growth = ['revenueGrowth','earningsGrowth','earningsQuarterlyGrowth','revenueQuarterlyGrowth','heldPercentInsiders']
value = ['priceToBook','enterpriseValue','enterpriseToRevenue','enterpriseToEbitda','trailingEps','priceToSalesTrailing12Months','trailingPE']
business = ['returnOnAssets','returnOnEquity','grossMargins','operatingMargins','profitMargins']
finance = ['debtToEquity','operatingCashflow','freeCashflow','totalCashPerShare','currentRatio','quickRatio','totalCash','totalDebt','BPS']
performance = ['totalRevenue','grossProfits','revenuePerShare','ebitdaMargins','EBITDAPS']
volitality = ['marketCap','currentPrice','fiftyDayAverage','twoHundredDayAverage','52WeekChange','ytdReturn','fiveYearAverageReturn','beta']


# 앱 로고
# from PIL import Image
# image = Image.open(f"{DATA_PATH}logo.png") # 무료 로고이미지
# st.sidebar.image(image, width=180) # use_column_width="auto" # 크기 사이드바에 맞추기

# 이모티콘 떨어지기(삭제할 예정)
# from streamlit_extras.let_it_rain import rain

# rain(
#     emoji="↗",
#     font_size=24, # 이모티콘 크기
#     falling_speed=6, # 떨어지는 속도
#     animation_length="infinite", 
# )


# 메인화면에 표시할 부분 (st.)
from streamlit_extras.colored_header import colored_header

colored_header(
    label='투자성향에 맞는 포트폴리오 추천',
    description="고객님의 투자 성향에 맞는 포트폴리오를 간편하게 추천해드립니다.",
    color_name="blue-70",
)
# st.title('투자성향에 맞는 포트폴리오 추천')
# st.divider()



# <<<<< 사이드바 >>>>>
st.sidebar.title('포트폴리오 선택')


# 주식 섹터에 대한 설명
# 세션 상태 초기화
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

# 사이드바에서 투자 성향 선택
selected_item = st.sidebar.radio("(1) 당신의 투자 성향을 선택하세요:", ('안정형', '중립형', '수익형'))

# 투자 성향에 따른 포트폴리오 제공
# 1.원하는 기업 하나 선택(소비자가 선택하지 않으면 어려운 숙제)
# 2.그대로 활용 vs 기대수익에 맞게 구성
# 3.섹터를 선택하지 않았을 때 오류메세지 문제(해결완료)
# 4.안정형을 먼저 완성하고 반복문에 코드를 수정하여 중립형과 수익형 코드 작성


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
    selected_stock = [] # 변수 초기화
    tmp_stock = pd.DataFrame()  # 변수 초기화

    multi_select = st.sidebar.multiselect('Sector를 2개 이상 선택하세요.', list(labelling.keys()), 
                                          max_selections=3, placeholder='Sector를 선택하세요.') 

    # multi_select 조건문
    st.write('선택한 Sector:', multi_select)

    if multi_select:
        conditions = [] # 변수 초기화
        for idx, selected in enumerate(multi_select):
            condition = data[selected].isin(labelling[selected])
            conditions.append(condition)
        
        if conditions:
            recommendation = data[np.logical_and.reduce(conditions)] # or조건은 np.logical_or.reduce 사용
            selected_stock = recommendation["Name"].to_list()
            tmp_stock = tmp[selected_stock]
            st.write('추천 종목:', selected_stock)

        # 포트폴리오 시각화 코드 시작
        if selected_stock:
            selected_stock = st.sidebar.selectbox("목록 중 원하는 기업을 선택하세요.", selected_stock)

            if selected_stock in tmp_stock.columns:
                stocks = list(tmp_stock.corr()[[selected_stock]].sort_values(by=selected_stock)[:4].index)+[selected_stock] # 상위 4개 선정
                st.write('선택된 기업과 상위 4개의 기업 포트폴리오')
                st.dataframe(stocks)

                # 수익률과 공분산 구하기
                daily_ret = tmp[stocks].pct_change()
                annual_ret = (1+daily_ret.mean())**tmp[stocks].shape[0]-1
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

                # 기대 수익 포트폴리오
                # 원하는 기대 수익은 얼마인가?
                exp_ret = st.sidebar.slider('원하는 기대 수익', min_value=0.5, max_value=15.0, step=0.1)/100

                w = sympy.Symbol('w')
                equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret

                solution = sympy.solve(equation, w)
                solution = float(solution[0])


                # 수평선 표시
                st.divider()

                # 기대수익을 위한 포트폴리오 시작
                if solution >= 0:
                    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], 
                                        subplot_titles=("<b>포트폴리오", "<b>기대수익을 위한 포트폴리오"))
                    
                    # 기존 포트폴리오
                    fig.add_trace(go.Pie(
                        values=list(max_shape.values[0][-len(stocks):]),
                        labels=list(max_shape.columns[-len(stocks):]),
                        domain=dict(x=[0, 0.5]),
                        name="기존 포트폴리오"),
                        row=1, col=1)
                    
                        # 채권을 포함한 포트폴리오

                        # # 기존 주식 비중
                        # stock_values = list(max_shape.values[0][-len(stocks):] * (1 - float(solution)))
                        # # 채권 비중 추가
                        # stock_values.append(float(solution))
                        # # 레이블
                        # labels = list(max_shape.columns[-len(stocks):]) + ['채권']

                        # fig.add_trace(go.Pie(
                        #     values=stock_values,
                        #     labels=labels,
                        #     domain=dict(x=[0.5, 1.0]),
                        #     name="기대수익 포트폴리오"),
                        #     row=1, col=2)


                    # 기존 주식 비중 조정 (채권 비중을 고려하여)
                    stock_values = list(max_shape.values[0][-len(stocks):] * (1 - solution))
                    # 채권 비중 추가
                    stock_values.append(float(solution))
                    # 레이블
                    labels = list(max_shape.columns[-len(stocks):]) + ['채권']

                    fig.add_trace(go.Pie(
                        values=stock_values,
                        labels=labels,
                        domain=dict(x=[0.5, 1.0]),
                        name="기대수익 포트폴리오"),
                        row=1, col=2)




                    # # 기존 포트폴리오
                    # fig.add_trace(go.Pie(
                    #     values=list(max_shape.values[0][-len(stocks):]),
                    #     labels=list(max_shape.columns[-len(stocks):]),
                    #     domain=dict(x=[0, 0.5]),
                    #     name="GHG Emissions"),
                    #     row=1, col=1)
                    
                    # # 채권을 추가한 포트폴리오
                    # fig.add_trace(go.Pie(
                    #     values=list(max_shape.values[0][-len(stocks):] * (1 - float(solution))) + [float(solution)],
                    #     labels=list(max_shape.columns[-len(stocks):]) + ['채권'],
                    #     domain=dict(x=[0.5, 1.0]),
                    #     name="CO2 Emissions"),
                    #     row=1, col=2)
                    


                    # 그래프 출력
                    st.plotly_chart(fig)

                else:
                    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], 
                                        subplot_titles=("<b>포트폴리오", f"<b>투자금 비중</b><br><sup>자기자본의 {-solution*100:0.4}%만큼 차입</sup>"))
                    
                    # 기존 포트폴리오
                    fig.add_trace(go.Pie(
                        values=list(max_shape.values[0][-len(stocks):]),
                        labels=list(max_shape.columns[-len(stocks):]),
                        domain=dict(x=[0, 0.5])),
                        row=1, col=1)
                    
                    # 차입금을 포함한 포트폴리오
                    fig.add_trace(go.Pie(
                        values=[1 / (1 - solution), 1 - (1 / (1 - solution))],
                        labels=['자기자본', '차입금'],
                        domain=dict(x=[0.5, 1.0])),
                        row=1, col=2)
                    
                    # 그래프 출력
                    st.plotly_chart(fig)

                st.write(f"채권의 비중 : {solution}")
                st.write(f"이 경우 Risk : {(1 - solution) * max_shape['Risk'].iloc[0]}")


                # <<<<<기존 코드>>>>>
                # if solution >= 0:
                    
                #     fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", "<b>기대수익을 위한 포트폴리오"))


                #     fig.add_trace(go.Pie(
                #         values=list(max_shape.values[0][-3:]),
                #         labels=list(max_shape.columns[-3:]),
                #         domain=dict(x=[0, 0.5]),
                #         name="GHG Emissions"),
                #         row=1, col=1)

                #     fig.add_trace(go.Pie(
                #         values=list(max_shape.values[0][-3:]* (1-float(solution)))+[float(solution)] ,
                #         labels=list(max_shape.columns[-3:]) + ['채권'],
                #         domain=dict(x=[0.5, 1.0]),
                #         name="CO2 Emissions"),
                #         row=1, col=2)

                #     # 그래프 출력
                #     st.plotly_chart(fig)

                # else:
                #     fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("<b>포트폴리오", f"<b>투자금 비중</b><br><sup>자기자본의 {-solution*100:0.4}%만큼 차입</sup>"))


                #     fig.add_trace(go.Pie(
                #         values=list(max_shape.values[0][-3:]),
                #         labels=list(max_shape.columns[-3:]),
                #         domain=dict(x=[0, 0.5])),
                #         row=1, col=1)

                #     fig.add_trace(go.Pie(
                #         values=[1/(1-solution),1-(1/(1-solution))] ,
                #         labels=['자기자본','차입금'],
                #         domain=dict(x=[0.5, 1.0])),
                #         row=1, col=2)

                #     # 그래프 출력
                #     st.plotly_chart(fig)


                # st.write(f"채권의 비중 : {solution}")
                # st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

    else:
        st.write("선택된 섹터가 없습니다. 섹터를 선택해주세요.")


elif selected_item == "중립형":
    st.write("중립형은 추후 추가될 예정입니다.")



elif selected_item == "수익형":
    selected_stock = []  # 변수 초기화
    tmp_stock = pd.DataFrame()  # 변수 초기화

    multi_select = st.sidebar.multiselect('Sector를 2개 이상 선택하세요.', list(labelling.keys()), 
                                          max_selections=3, placeholder='Sector를 선택하세요.')

    # multi_select 조건문
    st.write('선택한 Sector:', multi_select)

    if multi_select:
        conditions = []  # 변수 초기화
        for idx, selected in enumerate(multi_select):
            condition = data[selected].isin(labelling[selected])
            conditions.append(condition)

        if conditions:
            recommendation = data[np.logical_and.reduce(conditions)]
            selected_stock = recommendation["Name"].to_list()
            tmp_stock = tmp[selected_stock]
            st.write('추천 종목:', selected_stock)

        # 포트폴리오 시각화 코드 시작
        if selected_stock:
            # 수익률과 공분산 구하기
            daily_ret = tmp[selected_stock].pct_change()
            annual_ret = daily_ret.mean() * tmp[selected_stock].shape[0]
            daily_cov = daily_ret.cov()
            annual_cov = daily_cov * tmp[selected_stock].shape[0]

            # Shape 값 계산
            shape_values = (annual_ret - 0.02) / daily_ret.std() * np.sqrt(252)
            # 상위 5개 종목 선택
            top_shape = shape_values.sort_values(ascending=False)[:5]  # 상위 5개
            stocks = top_shape.index.tolist()

            # annual_cov와 annual_ret 필터링
            annual_cov = annual_cov.loc[stocks, stocks]
            annual_ret = annual_ret[stocks]

            st.write('개별 Shape 상위 5개의 기업 포트폴리오')
            st.dataframe(stocks)



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

            # 기대 수익 포트폴리오
            # 원하는 기대 수익은 얼마인가?
            exp_ret = st.sidebar.slider('원하는 기대 수익', min_value=0.5, max_value=50.0, step=0.1) / 100

            w = sympy.Symbol('w')
            equation = w * 0.02 + (1 - w) * max_shape['Returns'].values[0] - exp_ret

            solution = sympy.solve(equation, w)
            solution = float(solution[0])


            # 수평선 표시
            st.divider()

            # 기대수익을 위한 포트폴리오 시작
            if solution >= 0:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], 
                                    subplot_titles=("<b>포트폴리오", "<b>기대수익을 위한 포트폴리오"))
                
                # 기존 포트폴리오
                fig.add_trace(go.Pie(
                    values=list(max_shape.values[0][-len(stocks):]),
                    labels=list(max_shape.columns[-len(stocks):]),
                    domain=dict(x=[0, 0.5]),
                    name="기존 포트폴리오"),
                    row=1, col=1)
                
                # 채권을 포함한 포트폴리오
                # 기존 주식 비중 조정 (채권 비중을 고려하여)
                stock_values = list(max_shape.values[0][-len(stocks):] * (1 - solution))
                # 채권 비중 추가
                stock_values.append(float(solution))
                # 레이블
                labels = list(max_shape.columns[-len(stocks):]) + ['채권']

                fig.add_trace(go.Pie(
                    values=stock_values,
                    labels=labels,
                    domain=dict(x=[0.5, 1.0]),
                    name="기대수익 포트폴리오"),
                    row=1, col=2)

                
                # 그래프 출력
                st.plotly_chart(fig)



                # # 기존 포트폴리오
                # fig.add_trace(go.Pie(
                #     values=list(max_shape.values[0][-len(stocks):]),
                #     labels=list(max_shape.columns[-len(stocks):]),
                #     domain=dict(x=[0, 0.5]),
                #     name="GHG Emissions"),
                #     row=1, col=1)
                
                # # 채권을 추가한 포트폴리오
                # fig.add_trace(go.Pie(
                #     values=list(max_shape.values[0][-len(stocks):] * (1 - float(solution))) + [float(solution)],
                #     labels=list(max_shape.columns[-len(stocks):]) + ['채권'],
                #     domain=dict(x=[0.5, 1.0]),
                #     name="CO2 Emissions"),
                #     row=1, col=2)
                
                # 그래프 출력
                # st.plotly_chart(fig)

            else:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], 
                                    subplot_titles=("<b>포트폴리오", f"<b>투자금 비중</b><br><sup>자기자본의 {-solution*100:0.4}%만큼 차입</sup>"))
                
                # 기존 포트폴리오
                fig.add_trace(go.Pie(
                    values=list(max_shape.values[0][-len(stocks):]),
                    labels=list(max_shape.columns[-len(stocks):]),
                    domain=dict(x=[0, 0.5])),
                    row=1, col=1)
                
                # 차입금을 포함한 포트폴리오
                fig.add_trace(go.Pie(
                    values=[1 / (1 - solution), 1 - (1 / (1 - solution))],
                    labels=['자기자본', '차입금'],
                    domain=dict(x=[0.5, 1.0])),
                    row=1, col=2)
                
                # 그래프 출력
                st.plotly_chart(fig)

            st.write(f"채권의 비중 : {solution}")
            st.write(f"이 경우 Risk : {(1 - solution) * max_shape['Risk'].iloc[0]}")


else:
    st.write("선택된 섹터가 없습니다. 섹터를 선택해주세요.")

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
# from bardapi import Bard
# import bardapi
# import os
# os.environ["_BARD_API_KEY"] = ""

# if checkbox_btn:
    # prompt = st.chat_input("bard api와 채팅하기")
    # if prompt:
    #     response = bardapi.core.Bard().get_answer(prompt)
    #     st.write(f"bard: {response['content']}")


        # def bard_api(prompt):
        #     response = bardapi.core.Bard().get_answer(prompt)
        #     return response['content']
    
        
    # from streamlit_extras.stateful_chat import chat, add_message
    # with chat(key="my_chat"):
    #     if prompt := st.chat_input():
    #         add_message("user", prompt, avatar="🧑‍💻")
    #         response = bardapi.core.Bard().get_answer(prompt)
    #         add_message("assistant", "bard api: ", response['content'], avatar="🦜")

