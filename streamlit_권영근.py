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
DATA_PATH = "C:/python-code/03_data_collection_management/final/"
data = pd.read_csv(f"{DATA_PATH}labeled_data_final.csv")


def main():
    st.title("페이지 선택 예제")
    selected_item = st.radio("당신의 투자 성향을 선택하세요!", ('안정형', '수익형'))
    
    
    if selected_item == "안정형":
        st.write("안정형을 선택하셨습니다.")
        page_1()
    
    elif selected_item == "수익형":
        st.write("수익형을 선택하셨습니다.")
        page_2()

def page_1():
    st.header("안정형 포트폴리오")
    
    multi_select = st.multiselect('7개의 섹터 중 3개의 섹터를 선택하세요!',
                                  ['business', 'dividend', 'finance', 'growth', 'performance', 'value', 'volitality'],
                                  key= 'multiselect'
)
    if len(multi_select) >3:
        st.warning("3개의 항목까지만 선택할 수 있습니다. 최초 선택한 3개만 인정합니다.")
        multi_select = multi_select[:3]
    elif len(multi_select) <3:
        st.warning("3개의 항목을 선택하세요. 3개를 선택해야지만 결과를 확인할 수 있습니다.")
        multi_select = None
        
        
    st.write('당신의 선택:', multi_select)
    
    if multi_select[0]=='business':
        a= ((data['business']=='A')|(data['business']=='B'))
    elif multi_select[0]=='dividend':
        a= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[0]=='finance':
        a= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[0]=='growth':
        a= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[0]=='performance':
        a= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[0]=='value':
        a= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[0]=='volitality':
        a= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))

    if multi_select[1]=='business':
        b= ((data['business']=='A')|(data['business']=='B'))
    elif multi_select[1]=='dividend':
        b= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[1]=='finance':
        b= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[1]=='growth':
        b= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[1]=='performance':
        b= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[1]=='value':
        b= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[1]=='volitality':
        b= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))
            
            
    if multi_select[2]=='business':
        c= ((data['business']=='A')|(data['business']=='B'))
    elif multi_select[2]=='dividend':
        c= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[2]=='finance':
        c= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[2]=='growth':
        c= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[2]=='performance':
        c= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[2]=='value':
        c= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[2]=='volitality':
        c= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))

    
    recommendation= data[((a)&(b)&(c))]
    stocks = recommendation["Name"].to_list()
    st.write('추천 종목:', stocks)   





def page_2():
    st.header("수익형 포트폴리오")
    multi_select = st.multiselect('7개의 섹터 중 3개의 섹터를 선택하세요!',
                                  ['business', 'dividend', 'finance', 'growth', 'performance', 'value', 'volitality'],
                                  key= 'multiselect'
    )
    
    
    if len(multi_select) >3:
        st.warning("3개의 항목까지만 선택할 수 있습니다. 최초 선택한 3개만 인정합니다.")
        multi_select = multi_select[:3]
    elif len(multi_select) <3:
        st.warning("3개의 항목을 선택하세요. 3개를 선택해야지만 결과를 확인할 수 있습니다.")
        multi_select = None

    st.write('당신의 선택:', multi_select)
    
    if multi_select[0]=='business':
        a= ((data['business']=='A')|(data['finance']=='B'))
    elif multi_select[0]=='dividend':
        a= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[0]=='finance':
        a= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[0]=='growth':
        a= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[0]=='performance':
        a= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[0]=='value':
        a= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[0]=='volitality':
        a= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))

    if multi_select[1]=='business':
        b= ((data['business']=='A')|(data['finance']=='B'))
    elif multi_select[1]=='dividend':
        b= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[1]=='finance':
        b= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[1]=='growth':
        b= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[1]=='performance':
        b= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[1]=='value':
        b= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[1]=='volitality':
        b= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))
            
            
    if multi_select[2]=='business':
        c= ((data['business']=='A')|(data['finance']=='B'))
    elif multi_select[2]=='dividend':
        c= ((data['dividend']=='A')|(data['dividend']=='B')|(data['dividend']=='C'))
    elif multi_select[2]=='finance':
        c= ((data['finance']=='A')|(data['finance']=='B')|(data['finance']=='C')|(data['finance']=='D')|(data['finance']=='E')|(data['finance']=='F')|(data['finance']=='G'))
    elif multi_select[2]=='growth':
        c= ((data['growth']=='A')|(data['growth']=='A-')|(data['growth']=='B')|(data['growth']=='B-'))
    elif multi_select[2]=='performance':
        c= ((data['performance']=='A+')|(data['performance']=='A')|(data['performance']=='B'))
    elif multi_select[2]=='value':
        c= ((data['value']=='A')|(data['value']=='B')|(data['value']=='C'))
    elif multi_select[2]=='volitality':
        c= ((data['volitality']=='A+')|(data['volitality']=='A')|(data['volitality']=='B')|(data['volitality']=='C'))


    recommendation= data[((a)&(b)&(c))]
    stocks = recommendation["Name"].to_list()
    st.write('추천 종목:', stocks)    
        
        



if __name__ == "__main__":
    main()