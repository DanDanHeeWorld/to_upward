import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from bardapi import Bard
import bardapi
import os
import requests
from streamlit_extras.colored_header import colored_header

from pages import shape
from pages import correlation
from pages import chatbot
from pages import chatbot2

if "page" not in st.session_state:
    st.session_state.page = "home"


os.environ["_BARD_API_KEY"] = "agiHSpjsPdok9qnUEsHtnYCjXQdTCyCVEqE52VfRwUKzrnpRPKNcjnT0GYAszwQ541vYZg."

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


# # 기본 구성
# # 'generated'와 'past' 키 초기화
# st.session_state.setdefault('generated', [])
# st.session_state.setdefault('past', [])

# def on_input_change():
#     # 사용자 질문
#     user_input = st.session_state.user_input
#     st.session_state.past.append(user_input)
#     # 바드 답변
#     bard = Bard(token=os.environ["_BARD_API_KEY"], token_from_browser=True, session=session, timeout=30)
#     response = bard.get_answer(user_input)
#     st.session_state.generated.append({"type": "normal", "data": response['content']})

# def on_btn_click():
#     del st.session_state.past[:]
#     del st.session_state.generated[:]


# st.title("Portfolio Chatbot")

# chat_placeholder = st.empty()

# with chat_placeholder.container():    
#     for i in range(len(st.session_state['generated'])):                
#         message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
#         message(
#             st.session_state['generated'][i]['data'], 
#             key=f"{i}", 
#             allow_html=True,
#             is_table=True if st.session_state['generated'][i]['type']=='table' else False
#         )
    
#     st.button("Clear message", on_click=on_btn_click)

# with st.container():
#     st.text_input("User Input:", on_change=on_input_change, key="user_input")




# 질문-답변 로직 구성
# 'generated'와 'past' 키 초기화
st.session_state.setdefault('generated', [{'type': 'normal', 'data': "어떤 종목에 투자하려고 하시나요?"}])
st.session_state.setdefault('past', ['주식 투자를 한번 해보고 싶은데, 어떻게 하면 될까?'])
st.session_state.setdefault('chat_stage', 1)

# st.session_state에 ({"type": "normal", "data": "실제 데이터(텍스트, 코드 등)"}
# type은 normal = 일반 텍스트, code = python 코드로 표시


colored_header(
    label='Portfolio Chatbot',
    description=None,
    color_name="blue-70",
)
# st.title("Portfolio Chatbot")

chat_placeholder = st.empty()


# def on_btn_click():
#     del st.session_state['past'][:]
#     del st.session_state['generated'][:]
#     st.session_state['chat_stage'] = 1

def on_btn_click():
    st.session_state['past'] = ['주식 투자를 한번 해보고 싶은데, 어떻게 하면 될까?']
    st.session_state['generated'] = [{'type': 'normal', 'data': "어떤 종목에 투자하려고 하시나요?"}]
    st.session_state['chat_stage'] = 1

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)

    if st.session_state['chat_stage'] == 1:
        st.session_state['stock_interest'] = user_input
        st.session_state['generated'].append({"type": "normal", "data": "어떤 sector에 대해서 궁금한가요?"})
        st.session_state['chat_stage'] = 2

    elif st.session_state['chat_stage'] == 2:
        st.session_state['sector_interest'] = user_input
        example_prompt = f"""궁금한 sector는 {st.session_state['sector_interest']}이며, 
                            관심있는 주식 종목은 {st.session_state['stock_interest']} 입니다.
                            
                            Name                                             삼성전자
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
                            추정PER                                          69.4
                            동종업계trailingPE                                9.59
                            priceToBook                                      5.44
                            enterpriseValue                135,256,086,675,456.00
                            enterpriseToRevenue                              4.51
                            enterpriseToEbitda                              34.66
                            trailingEps                                      3306
                            priceToSalesTrailing12Months                     4.44
                            trailingPE                                     167.27
                            예시와 비슷한 글을 생성해줘.
                            """
        
        bard = Bard(token=os.environ["_BARD_API_KEY"], token_from_browser=True, session=session, timeout=30)
        response = bard.get_answer(example_prompt)
        st.session_state['generated'].append({"type": "normal", "data": response['content']})
        st.session_state['chat_stage'] = 1

with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'],
            key=f"{i}",
            allow_html=True,
            is_table=True if st.session_state['generated'][i]['type'] == 'table' else False
        )
    
    st.button("Clear message", on_click=on_btn_click, key="clear_key")

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")


# message 함수 설명
# message (str) 
# 설명: 채팅 컴포넌트에 표시될 메시지 내용입니다.
# is_user (bool, 기본값: False)  
# 설명: 메시지의 발신자가 사용자인지 여부를 지정합니다. True로 설정하면 메시지가 오른쪽에 정렬됩니다.
# avatar_style (AvatarStyle 또는 None, 기본값: None) 
# 설명: 메시지 발신자의 아바타 스타일을 지정합니다. 기본값은 bottts (for non-user)과 pixel-art-neutral (for user)입니다. 아바타 스타일은 dicebear에서 선택할 수 있습니다.
# logo (str 또는 None, 기본값: None)
# 설명: 아바타 대신 사용할 로고를 지정합니다. 이는 챗봇에 브랜드를 부여하려는 경우 유용합니다.
# seed (int 또는 str, 기본값: 88)
# 설명: 사용할 아바타를 선택하는 데 사용되는 시드입니다.
# allow_html (bool, 기본값: False)
# 설명: 메시지에 HTML을 사용할 수 있는지 여부를 지정합니다. True로 설정하면 HTML 사용이 가능해집니다.
# is_table (bool, 기본값: False)
# 설명: 테이블에 특정 스타일을 적용할지 여부를 지정합니다.
# key (str 또는 None, 기본값: None)
# 설명: 이 컴포넌트를 고유하게 식별하는 선택적 키입니다. 이 값이 None이고 컴포넌트의 인수가 변경되면, 컴포넌트는 Streamlit 프론트엔드에서 다시 마운트되고 현재 상태를 잃게 됩니다.



# 이미지, 오디오 사용 참고용 코드
# audio_path = "https://docs.google.com/uc?export=open&id=16QSvoLWNxeqco_Wb2JvzaReSAw5ow6Cl"
# img_path = "https://www.groundzeroweb.com/wp-content/uploads/2017/05/Funny-Cat-Memes-11.jpg"
# youtube_embed = '''
# <iframe width="400" height="215" src="https://www.youtube.com/embed/LMQ5Gauy17k" title="YouTube video player" frameborder="0" allow="accelerometer; encrypted-media;"></iframe>
# '''

# markdown = """
# ### HTML in markdown is ~quite~ **unsafe**
# <blockquote>
#   However, if you are in a trusted environment (you trust the markdown). You can use allow_html props to enable support for html.
# </blockquote>

# * Lists
# * [ ] todo
# * [x] done

# Math:

# Lift($L$) can be determined by Lift Coefficient ($C_L$) like the following
# equation.

# $$
# L = \\frac{1}{2} \\rho v^2 S C_L
# $$

# ~~~py
# import streamlit as st

# st.write("Python code block")
# ~~~

# ~~~js
# console.log("Here is some JavaScript code")
# ~~~

# """

# table_markdown = '''
# A Table:

# | Feature     | Support              |
# | ----------: | :------------------- |
# | CommonMark  | 100%                 |
# | GFM         | 100% w/ `remark-gfm` |
# '''


# st.session_state.setdefault(
#     'past', 
#     ['plan text with line break',
#      'play the song "Dancing Vegetables"', 
#      'show me image of cat', 
#      'and video of it',
#      'show me some markdown sample',
#      'table in markdown']
# )
# st.session_state.setdefault(
#     'generated', 
#     [{'type': 'normal', 'data': 'Line 1 \n Line 2 \n Line 3'},
#      {'type': 'normal', 'data': f'<audio controls src="{audio_path}"></audio>'}, 
#      {'type': 'normal', 'data': f'<img width="100%" height="200" src="{img_path}"/>'}, 
#      {'type': 'normal', 'data': f'{youtube_embed}'},
#      {'type': 'normal', 'data': f'{markdown}'},
#      {'type': 'table', 'data': f'{table_markdown}'}]
# )