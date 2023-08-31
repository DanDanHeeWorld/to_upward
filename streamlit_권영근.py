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
import pandas as pd
from bs4 import BeautifulSoup
import requests


DATA_PATH = "C:/python-code/03_data_collection_management/final/"
data = pd.read_csv(f"{DATA_PATH}labeled_data_final.csv")
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
tmp = pd.DataFrame() 


def main():






    menu= ['포트폴리오', '기업 거래 정보']
    choice= st.sidebar.selectbox('메뉴', menu)

    if choice== '포트폴리오':


        st.title("투자성향에 맞는 포트폴리오 추천")
        selected_item = st.radio("당신의 투자 성향을 선택하세요!", ('안정형', '수익형'))
        
        if selected_item == "안정형":
            st.write("안정형을 선택하셨습니다.")
            page_1()
        
        elif selected_item == "수익형":
            st.write("수익형을 선택하셨습니다.")
            page_2()




    elif choice == '기업 거래 정보':
            st.title('간단한 기업 거래 정보')
            user_input= st.text_input("원하는 기업 이름 입력:", help='정보를 얻고자 하는 기업의 이름을 입력하세요.')
            if user_input in data['Name'].to_list():
                end = dt.datetime.today().date().strftime("%Y%m%d")
                start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
                stocks = user_input
                tmp = pd.DataFrame()
                tmp[stocks] = stock.get_market_ohlcv_by_date(start, end, data[data['Name'] == user_input]['Code'].iloc[0])['종가']
                ohlcv = stock.get_market_ohlcv_by_date(start, end, data[data['Name'] == user_input]['Code'].iloc[0])



                recent_ohlcv = st.slider("최근 며칠의 데이터를 보시겠습니까?", 7, 365, step=7)
                ohlcv = ohlcv.tail(recent_ohlcv)

                st.write(f'[{user_input} 에 대한 간단한 설명입니다.]')

                user_input2= data[data['Name']==user_input]['Code'].iloc[0]
                url = f'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{user_input2}&cID=&MenuYn=Y&ReportGB=&NewMenuID=11&stkGb=701'
                response = requests.get(url)
                html = response.text
                soup= BeautifulSoup(html, 'html.parser')


                my_id_element= soup.find_all(id="bizSummaryContent")

                for element in my_id_element:
                    cleaned_content = ' '.join(element.stripped_strings)
                    cleaned_content = BeautifulSoup(cleaned_content, 'html.parser').get_text()
                    cleaned_content
                
                st.subheader(user_input + f' 최근 {recent_ohlcv}일 가격 변화')
                st.line_chart({
                    '시가': ohlcv['시가'],
                    '고가': ohlcv['고가'],
                    '저가': ohlcv['저가'],
                    '종가': ohlcv['종가']
                })
                st.subheader(user_input + f' 최근 {recent_ohlcv}일 거래량')
                st.bar_chart(ohlcv['거래량'],color='#F1B9B7',width=10)
                ohlcv['등락률'] = ohlcv['종가'].pct_change() * 100
                st.subheader(user_input + f' 최근 {recent_ohlcv}일 등락률')
                st.line_chart(ohlcv['등락률'],color='#A3A6E5',width=20)


                data['Code']= data['Code'].apply(lambda x: str(x).zfill(6))
                
                

            else:
                st.warning('지원하지 않는 기업명입니다. 다시 입력해주세요.')


        

            
        
            
def page_1():
    st.header("안정형 포트폴리오")
    options_with_descriptions= {
        "business": "business sector는 영업이익률이나 순이익률을 높은 편인 그룹입니다.",
        "dividend": "dividend sector는 5년 평균 배당률이 높은 그룹입니다.",
        "finance": "finance sector는 당좌비율이나 유동비율이 높은 그룹입니다.",
        "growth": "growth sector는 연별성장이나 단기 성장성이 높은 그룹입니다.",
        "performance": "performance sector는 총 수익이나 주당 매출이 높은 그룹입니다.",
        "value": "value sector는 동종업계 대비 PER이 높은 그룹입니다.",
        "volitality": "volitality sector는 1년 평균 수익률이나 5년 평균 수익률이 높은 그룹입니다."
    }
    #options_with_descriptions2= {
        #"business": "business sector는 A/B/C/D 중 A 그룹만 채택하였습니다. (36개)",
        #"dividend": "dividend sector는 A/B/C/D/E 중 A/B 그룹만 채택하였습니다. (38개)",
        #"finance" : "finance sector는 A+/A/B/C/D/E/F/G/H/I 중 A+/A/B/C/D/E 그룹만 채택하였습니다. (21개)",
        #"growth": "growth sector는 A/A-/B/B-/C+/C/D 중 A/A- 그룹만 채택하였습니다. (16개)",
        #"performance": "performance sector는 A+/A/B/C/D/E 중 A+/A/B/C 그룹만 채택하였습니다. (36개)",
        #"value": "value sector는 A/B/C/D/E/F/G 중 A/B/C 그룹만 채택하였습니다. (22개)",
        #"volitality": "volitality sector는 A+/A/B 그룹만 채택하였습니다. (29개)"
    #}
    #st.write('business, dividend, finance, growth, performance, value, volitality')
    multi_select = st.multiselect('7개의 섹터 중 3개의 섹터를 선택하세요!',
                                options= list(options_with_descriptions.keys()),
                                format_func= lambda option: options_with_descriptions[option],
                                help= 'business, dividend, finance, growth, performance, value, volitality'
                                )
    if len(multi_select) >=3:
        if len(multi_select) >3:
            st.warning("3개의 항목까지만 선택할 수 있습니다. 최초 선택한 3개만 인정합니다.")
            multi_select = multi_select[:3]

  

        for option in multi_select:
            st.write(f'당신의 선택: {option}')
            st.write(f'{options_with_descriptions[option]}')
            #st.write(f'{options_with_descriptions2[option]}')
            
        if multi_select[0]=='business':
            a= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[0]=='dividend':
            a= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[0]=='finance':
            a= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[0]=='growth':
            a= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[0]=='performance':
            a= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[0]=='value':
            a= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[0]=='volitality':
            a= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))

        if multi_select[1]=='business':
            b= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[1]=='dividend':
            b= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[1]=='finance':
            b= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[1]=='growth':
            b= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[1]=='performance':
            b= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[1]=='value':
            b= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[1]=='volitality':
            b= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))
                
                
        if multi_select[2]=='business':
            c= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[2]=='dividend':
            c= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[2]=='finance':
            c= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[2]=='growth':
            c= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[2]=='performance':
            c= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[2]=='value':
            c= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[2]=='volitality':
            c= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))

        
        recommendation= data[((a)&(b)&(c))]
        stocks = recommendation["Name"].to_list()
        df= pd.DataFrame(recommendation)
        result_df=df[df.columns[:-7]]
        result_df= result_df.reset_index(drop=True)
        st.write('추천 종목:', result_df)


        #tmp= pd.DataFrame()
        #end = dt.datetime.today().date().strftime("%Y%m%d")
        #start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
        #for n in stocks:
            #tmp[n] = stock.get_market_ohlcv_by_date(start, end, data[data['Name'] == n]['Code'])['종가']
        #tmp.corr()[[f'{stocks[0]}']].sort_values(by=f'{stocks[0]}')

        st.write("이 중 기준으로 할 하나의 기업을 선택하세요.")

        results= []
        for i in range(len(stocks)):
            if stocks[i] is not None:
                results.append(stocks[i])
            else:
                pass
        if results:
            selected_result = st.selectbox('기준 선택', results ,help= '선택한 기업을 기준으로 상관계수가 낮은 순서대로 다른 기업들이 자동 선택됩니다.')
            st.write('선택한 결과: ', selected_result)

        else:
            st.write('아직 결과 값이 선택되지 않았습니다.')
        end = dt.datetime.today().date().strftime("%Y%m%d")
        start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
        tmp = pd.DataFrame()
        for n in stocks:
            tmp[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']
        d=tmp.corr()[[f'{selected_result}']].sort_values(by=f'{selected_result}')
        mask = d[f'{selected_result}']<0.5
        dd=tmp.corr()[[f'{selected_result}']][mask].sort_values(by=f'{selected_result}')
        dd
        stocks = list(dd.index)[0:3]+[f"{selected_result}"]# 상위 3개 선정
        st.write('상관계수 상위 3개와 선택한 기준:', stocks)
        daily_ret = tmp[stocks].pct_change()
        annual_ret = (1+daily_ret.mean())**tmp[stocks].shape[0]-1
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
            elif tmp2.loc[i, 'Returns'] >= best_ret:
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

        st.plotly_chart(fig)
        st.write('max_shape')
        st.dataframe(max_shape)
        st.write('min_risk')
        st.dataframe(min_risk)

        exp_ret = st.slider("기대수익 선택", 0.0, 100.0, step=0.1) /100

        condition= (exp_ret > max_shape['Returns'])
        has_greater_value= condition.any()
        if has_greater_value:
            st.warning("max_return 이상의 값은 선택할 수 없습니다. max_return 값을 선택한 것으로 간주합니다.")
            exp_ret= max_shape['Returns']
        else:
            pass
        
        


        import sympy
        w = sympy.Symbol('w')
        equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret
        
        solution = sympy.solve(equation, w)
        solution = float(solution[0])
        st.write(f"채권의 비중 : {solution}")
        st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

        if 1 > solution >= 0:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("포트폴리오", f"기대수익을 위한 포트폴리오자기자본의 {solution*100:0.4}%만큼 채권투자"))
            
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
        elif solution < 0:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("포트폴리오", f"투자금 비중자기자본의 {-solution*100:0.4}%만큼 차입"))
            
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
        else:
            st.warning('채권의 비중이 1 이상입니다. 기대 수익을 다시 선택해주세요.')



    elif len(multi_select) <3:
        st.write("3개의 항목을 선택하세요. 3개를 선택해야지만 결과를 확인할 수 있습니다.")
        
















def page_2():
    st.header("수익형 포트폴리오")
    options_with_descriptions= {
        "business": "business sector는 영업이익률이나 순이익률을 높은 편인 그룹입니다.",
        "dividend": "dividend sector는 5년 평균 배당률이 높은 그룹입니다.",
        "finance": "finance sector는 당좌비율이나 유동비율이 높은 그룹입니다.",
        "growth": "growth sector는 연별성장이나 단기 성장성이 높은 그룹입니다.",
        "performance": "performance sector는 총 수익이나 주당 매출이 높은 그룹입니다.",
        "value": "value sector는 동종업계 대비 PER이 높은 그룹입니다.",
        "volitality": "volitality sector는 1년 평균 수익률이나 5년 평균 수익률이 높은 그룹입니다."
    }
    #options_with_descriptions2= {
        #"business": "business sector는 A/B/C/D 중 A 그룹만 채택하였습니다. (36개)",
        #"dividend": "dividend sector는 A/B/C/D/E 중 A/B 그룹만 채택하였습니다. (38개)",
        #"finance" : "finance sector는 A+/A/B/C/D/E/F/G/H/I 중 A+/A/B/C/D/E 그룹만 채택하였습니다. (21개)",
        #"growth": "growth sector는 A/A-/B/B-/C+/C/D 중 A/A- 그룹만 채택하였습니다. (16개)",
        #"performance": "performance sector는 A+/A/B/C/D/E 중 A+/A/B/C 그룹만 채택하였습니다. (36개)",
        #"value": "value sector는 A/B/C/D/E/F/G 중 A/B/C 그룹만 채택하였습니다. (22개)",
        #"volitality": "volitality sector는 A+/A/B 그룹만 채택하였습니다. (29개)"
    #}

    multi_select = st.multiselect('7개의 섹터 중 3개의 섹터를 선택하세요!',
                                options= list(options_with_descriptions.keys()),
                                format_func= lambda option: options_with_descriptions[option],
                                help= 'business, dividend, finance, growth, performance, value, volitality'
                                )
    if len(multi_select) >=3:
        if len(multi_select) >3:
            st.warning("3개의 항목까지만 선택할 수 있습니다. 최초 선택한 3개만 인정합니다.")
            multi_select = multi_select[:3]

  

        for option in multi_select:
            st.write(f'당신의 선택: {option}')
            st.write(f'{options_with_descriptions[option]}')
            #st.write(f'{options_with_descriptions2[option]}')
            
        if multi_select[0]=='business':
            a= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[0]=='dividend':
            a= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[0]=='finance':
            a= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[0]=='growth':
            a= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[0]=='performance':
            a= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[0]=='value':
            a= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[0]=='volitality':
            a= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))

        if multi_select[1]=='business':
            b= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[1]=='dividend':
            b= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[1]=='finance':
            b= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[1]=='growth':
            b= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[1]=='performance':
            b= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[1]=='value':
            b= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[1]=='volitality':
            b= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))
                
                
        if multi_select[2]=='business':
            c= ((data['business']=='A')|(data['business']=='B'))
        elif multi_select[2]=='dividend':
            c= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
        elif multi_select[2]=='finance':
            c= ((data['finance']=='A+')|(data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D'))
        elif multi_select[2]=='growth':
            c= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
        elif multi_select[2]=='performance':
            c= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
        elif multi_select[2]=='value':
            c= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C')|(data['value']=='D'))
        elif multi_select[2]=='volitality':
            c= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B'))

        
        recommendation= data[((a)&(b)&(c))]
        stocks = recommendation["Name"].to_list()
        df= pd.DataFrame(recommendation)
        result_df=df[df.columns[:-7]]
        result_df= result_df.reset_index(drop=True)
        st.write('추천 종목:', result_df)
        
        tmp = pd.DataFrame()
        end = dt.datetime.today().date().strftime("%Y%m%d")
        start = (dt.datetime.today().date() - dt.timedelta(365)).strftime("%Y%m%d")
        for n in stocks:
            tmp[n] = stock.get_market_ohlcv(start, end, data[data['Name'] == n]['Code'])['종가']

        daily_ret = tmp[stocks].pct_change()
        annual_ret = (1+daily_ret.mean())**tmp[stocks].shape[0]-1
        daily_cov = daily_ret.cov()
        annual_cov = daily_cov * tmp[stocks].shape[0]
        tmp2 = pd.DataFrame((annual_ret-0.02)/daily_ret.std()*np.sqrt(252),columns= ['Shape']).sort_values(by='Shape',ascending=False)
        tmp2


        stocks = list(tmp2.iloc[0:4].index)
        st.write('상위 4개 기업 자동 선택:', stocks)
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
            elif tmp2.loc[i, 'Returns'] >= best_ret :
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
        
        st.plotly_chart(fig)
        st.write('max_shape')
        st.dataframe(max_shape)
        st.write('min_risk')
        st.dataframe(min_risk)

        exp_ret = st.slider("기대수익 선택", 0.0, 100.0, step=0.1) /100

        condition= (exp_ret > max_shape['Returns'])
        has_greater_value= condition.any()
        if has_greater_value:
            st.warning("max_return 이상의 값은 선택할 수 없습니다. max_return 값을 선택한 것으로 간주합니다.")
            exp_ret= max_shape['Returns']
        else:
            pass
        
        


        import sympy
        w = sympy.Symbol('w')
        equation = w*0.02 + (1-w)*max_shape['Returns'].values[0] - exp_ret
        
        solution = sympy.solve(equation, w)
        solution = float(solution[0])
        st.write(f"채권의 비중 : {solution}")
        st.write(f"이 경우 Risk : {(1-solution)*max_shape['Risk'].iloc[0]}")

        if 1 > solution >= 0:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("포트폴리오", f"기대수익을 위한 포트폴리오자기자본의 {solution*100:0.4}%만큼 채권투자"))
            
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
        elif solution < 0:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],subplot_titles=("포트폴리오", f"투자금 비중자기자본의 {-solution*100:0.4}%만큼 차입"))
            
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
        else:
            st.warning('채권의 비중이 1 이상입니다. 기대 수익을 다시 선택해주세요.')



    elif len(multi_select) <3:
        st.write("3개의 항목을 선택하세요. 3개를 선택해야지만 결과를 확인할 수 있습니다.")
        


if __name__ == "__main__":
    main()
