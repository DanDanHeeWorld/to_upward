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

# import warnings
# warnings.filterwarnings('ignore')



# streamlit에 표시할 부분

# 타이틀 설정
st.title('투자성향에 맞는 포트폴리오 추천')


# <<<<< 사이드바 >>>>>
# 투자성향 입력 받기 # sidebar 추가 시 사이드바에서 선택
investment_style = st.sidebar.selectbox(
    '당신의 투자 성향을 선택하세요:',
    ['안정형', '중립형', '수익형']
)

# 사이드바 radio
with st.sidebar:
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.disabled = st.checkbox("Disable radio widget")
        st.session_state.horizontal = st.checkbox("Orient radio options horizontally")

    with col2:
        visibility = st.radio(
            "Set label visibility 👇",
            ["visible", "hidden", "collapsed"],
        )
        st.session_state.visibility = visibility if visibility != "hidden" else "visible"



# 투자성향에 따른 설명 제공
st.write(f'당신은 {investment_style} 투자자입니다.')

# 포트폴리오 추천 섹션
st.header('맞춤형 포트폴리오 추천')
st.write(f"다음은 당신의 투자 성향에 맞는 '{investment_style}' 포트폴리오입니다.")



# 여기에 실제 데이터 로드 및 처리 코드를 추가하기


# 마감일, 시작일
end = dt.datetime.today().date().strftime("%Y%m%d")
start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")

data = pd.DataFrame({
    'Name': ['삼성전자', 'LG에너지솔루션', '삼성바이오로직스', 'SK하이닉스', 'POSCO홀딩스'],
    'Code': ['005930', '051910', '207940', '000660', '005490']  # 종목 코드
})

stocks = ['삼성전자', 'LG에너지솔루션', '삼성바이오로직스', 'SK하이닉스', 'POSCO홀딩스']

# stocks = ['005930', '051910', '207940', '000660', '005490']
# 데이터 불러오기는 가능하지만 get_market_ohlcv_by_date에서 오류 발생(ipynb에서 정상 실행됨)
# DATA_PATH = "C:/Users/Jonghyeon/Desktop/파이널프로젝트/data/"
# data = pd.read_csv(f"{DATA_PATH}preprocessing_data5.csv")

# for n in stocks:
#     tmp[n] = stock.get_market_ohlcv_by_date(start, end, n)['종가']


# 아래 포트폴리오에서 사용할 데이터 가져오기
tmp = pd.DataFrame()

for n in stocks:
    tmp[n] = stock.get_market_ohlcv_by_date(start, end, data[data['Name'] == n]['Code'].iloc[0])['종가']


# 데이터프레임 추가기능 테스트(삭제 or 수정예정)
for n in stocks:
    ohlcv = stock.get_market_ohlcv_by_date(start, end, data[data['Name'] == n]['Code'].iloc[0])
    ohlcv = ohlcv.tail(30)  # 최근 30일의 데이터만 선택

    # 그래프 그리기
    st.subheader(n + ' 최근 30일 가격 변화')
    st.line_chart({
        '시가': ohlcv['시가'],
        '고가': ohlcv['고가'],
        '저가': ohlcv['저가'],
        '종가': ohlcv['종가']
    })
    st.subheader(n + ' 최근 30일 거래량')
    st.bar_chart(ohlcv['거래량'])

    # 등락률 계산 후 그래프 그리기
    ohlcv['등락률'] = ohlcv['종가'].pct_change() * 100
    st.subheader(n + ' 최근 30일 등락률')
    st.line_chart(ohlcv['등락률'])



# <<<<< 상관계수를 활용한 포트폴리오 >>>>>

daily_ret = tmp.pct_change()
# 5종목 1년간 수익률 평균(1년 평균 개장일=252)
annual_ret = daily_ret.mean() * 252

# 5종목 연간 리스크 = cov()함수를 이용한 일간변동률 의 공분산
daily_cov = daily_ret.cov()
# 5종목 1년간 리스크(1년 평균 개장일=252)
annual_cov = daily_cov * 252


st.subheader('상관계수를 활용한 상위 3개 기업 포트폴리오')
stocks = list(tmp.corr()[['LG에너지솔루션']].sort_values(by='LG에너지솔루션')[:2].index)+["LG에너지솔루션"] # 상위 3개 선정
daily_ret = tmp[stocks].pct_change()
annual_ret = daily_ret.mean() * tmp[stocks].shape[0]

daily_cov = daily_ret.cov()
annual_cov = daily_cov * tmp[stocks].shape[0]

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
  else:
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

st.write('max_shape')
st.dataframe(max_shape)
st.write('min_risk')
st.dataframe(min_risk)

# 수평선 표시
st.divider()

# <<<<< 개별 Shape 사용 >>>>>
st.subheader('개별 Shape을 사용한 상위 3개 기업 포트폴리오')
tmp2 = pd.DataFrame((annual_ret-0.02)/daily_ret.std()*np.sqrt(252),columns= ['Shape']).sort_values(by='Shape',ascending=False)
stocks = list(tmp2.iloc[0:3].index) # 상위 3개 선정

daily_ret = tmp[stocks].pct_change()
annual_ret = daily_ret.mean() * tmp[stocks].shape[0]

daily_cov = daily_ret.cov()
annual_cov = daily_cov * tmp[stocks].shape[0]

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
  else:
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

st.write('max_shape')
st.dataframe(max_shape)
st.write('min_risk')
st.dataframe(min_risk)

# 원하는 기대 수익은 얼마인가?
exp_ret = float(40/100) # 나중에 고객이 입력한 기대 수익으로 계산하기

w = sympy.Symbol('w')

equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret

solution = sympy.solve(equation, w)
solution = float(solution[0])

# print(f"채권의 비중 : {solution}")
# print(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

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

  fig.show()

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
st.plotly_chart(fig) # fig.show()대신 이 코드로 출력


# 추가 정보 제공
st.write('추가적인 분석이나 레포트는 추후 업데이트 예정입니다.')