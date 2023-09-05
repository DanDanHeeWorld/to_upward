import streamlit as st
import os
import requests
st.title("BardAPI for Python")


_BARD_API_KEY='agh_zsft7DZHNdcQzHRPqKYZSz1PxPLgjXBjJi0WoLvozgy1VGr73w7TO7nZjlcafyULaQ.'
os.environ['_BARD_API_KEY']='agh_zsft7DZHNdcQzHRPqKYZSz1PxPLgjXBjJi0WoLvozgy1VGr73w7TO7nZjlcafyULaQ.'
from bardapi import Bard
session = requests.Session()
session.headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
session.cookies.set("__Secure-1PSID", os.getenv("_BARD_API_KEY")) 
with st.container () :

    with st.spinner ( 'Wait till Bard gets the answers...' ) :
        input_text = """Name                                             삼성전자
추정PER                                            45.2
동종업계trailingPE                                   9.59
priceToBook                                      1.09
enterpriseValue                374,294,588,686,336.00
enterpriseToRevenue                              1.38
enterpriseToEbitda                               4.62
trailingEps                                      8057
priceToSalesTrailing12Months                     1.67
trailingPE                                       8.55
이러한 contents가 주어졌을 때, 다음과 같은 글을 생성해야한다.
삼성전자는 전문가들이 추정한 추정 PER가 실제 PER보다 매우 높은 것으로 보아, 현재 저평가 상태로 추정된다.
또, priceToBook가 1이상으로 안정적이며, enterpriseValue는 374,294,588,686,336으로 기업의 가치 또한 매우크다.
enterpriseToEbitda를 보았을 떄 기업의 가치 대비 4배이상의 수익을 내고 있는 것으로 보아 영업성과 또한 매우 우수하다.
또한, 동종업계 대비 PER가 낮은 것으로 보아 동종업계 내에서도 다소 저평가 중인 상태인 것을 알 수 있다.
종합적으로 보았을 때, 삼성전자는 영업성과는 뛰어나지만 저평가된 상태로 향후 주가상승을 기대할 수 있을 것으로 예상된다.

Name                                             LG에너지솔루션
추정PER                                            69.4
동종업계trailingPE                                   9.59
priceToBook                                      5.44
enterpriseValue                135,256,086,675,456.00
enterpriseToRevenue                              4.51
enterpriseToEbitda                              34.66
trailingEps                                      3306
priceToSalesTrailing12Months                     4.44
trailingPE                                     167.27
예시와 비슷한 글을 생성해줘."""

        try :
            bard = Bard(token_from_browser=True, session=session, timeout=30)
            response = bard.get_answer ( input_text ) [ 'content' ]
            # response = Bard.core.Bard().get_answer(input_text)
            st.write ( response )
            response = bard.get_answer ( "trailingPE란 무엇인가?" ) [ 'content' ]
            # response = Bard.core.Bard().get_answer(input_text)
            st.write ( response )
        except :
            st.error ( 'Please check if you have set the Cookie ' )