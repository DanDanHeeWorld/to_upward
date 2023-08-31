
import streamlit as st

# Survey Part
st.title("우상향, 나의 투자 포트폴리오")
st.sidebar.title("설문조사")

type_of_user = st.sidebar.radio("당신의 투자 성향은..?", ["수익형", "안정형"])

sectors = ["info", "trading", "price", "performance", "finance", "business", "value", "dividend", "growth"]
selected_sectors = st.sidebar.multiselect("중요하게 여기는 기업의 가치 섹터 3개를 고르세요:", sectors)
exp_ret = st.sidebar.number_input("원하는 기대수익률은 얼마입니까? (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1) / 100

if len(selected_sectors) > 3:
    st.warning("3개 이상의 섹터를 선택해주세요.")
elif len(selected_sectors) == 3:
    st.success(f"선택하신 섹터는 {', '.join(selected_sectors)} 입니다.")


shape_code_section = '''{{{}}}'''
# Variables
correlation_code_full_section = '''[correlation_code_full_section의 실제 내용]'''
shape_code_section = '''[shape_code_section의 실제 내용]'''

# ... [생략된 코드]

# Portfolio Part based on user's choice
if len(selected_sectors) == 3:
    if type_of_user == "안정형":
        # Code for correlation-based portfolio
        st.write("상관계수를 활용하는 포트폴리오 구성:")
        exec(correlation_code_full_section)
    elif type_of_user == "수익형":
        # Code or function for shape-based portfolio
        st.write("개별 shape를 활용하는 포트폴리오 구성:")
        st.code(shape_code_section)


# Portfolio Part based on user's choice
if len(selected_sectors) == 3:
    if type_of_user == "안정형":
        # Code for correlation-based portfolio
        st.write("상관계수를 활용하는 포트폴리오 구성:")
        exec(correlation_code_full_section)
    elif type_of_user == "수익형":
        # Code or function for shape-based portfolio
        st.write("개별 shape를 활용하는 포트폴리오 구성:")
        st.code(shape_code_section)


# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import random
import os

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

DATA_PATH = "./"
SEED = 42

from pykrx import stock

import plotly.express as px

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(f"{DATA_PATH}labeled_data_final.csv")
value_df = pd.read_csv(f"{DATA_PATH}value_df2.csv")
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


import datetime as dt

end = dt.datetime.today().date().strftime("%Y%m%d")
start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
# end = dt.datetime(2022,12,31).strftime("%Y%m%d")
# start = (dt.datetime(2022,12,31) - dt.timedelta(180)).strftime("%Y%m%d")
print(start,end)

# KOSPI 시가총액 5위
stocks = ['삼성전자', 'LG에너지솔루션', '삼성바이오로직스', 'SK하이닉스', 'POSCO홀딩스']

tmp = pd.DataFrame()
for n in stocks:
  tmp[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']


daily_ret = tmp.pct_change()
annual_ret = daily_ret.mean() * tmp.shape[0]

daily_cov = daily_ret.cov()
annual_cov = daily_cov * tmp.shape[0]

""" 상관계수를 활용"""

tmp.corr()[['LG에너지솔루션']].sort_values(by='LG에너지솔루션')[:2]

stocks = list(tmp.corr()[['LG에너지솔루션']].sort_values(by='LG에너지솔루션')[:2].index)+["LG에너지솔루션"]# 상위 3개 선정

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

import plotly.graph_objects as go


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


st.write(fig)

max_shape

min_risk

"""# 개별 Shape 사용"""

tmp2 = pd.DataFrame((annual_ret-0.02)/daily_ret.std()*np.sqrt(252),columns= ['Shape']).sort_values(by='Shape',ascending=False)


proprocessing_data = data

proprocessing_data['value'] = value_df['cluster']

proprocessing_data[(proprocessing_data['value'] == "A")|(proprocessing_data['dividend'] == "A")]

stocks = list(tmp2.iloc[0:3].index)# 상위 3개 선정

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

import plotly.graph_objects as go


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


st.write(fig)

max_shape

min_risk



import sympy

w = sympy.Symbol('w')

equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret

solution = sympy.solve(equation, w)
solution = float(solution[0])
print(f"채권의 비중 : {solution}")
print(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

from plotly.subplots import make_subplots
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

  st.write(fig)

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

  st.write(fig)

