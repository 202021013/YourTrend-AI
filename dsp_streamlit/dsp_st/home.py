import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="과제 도우미",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 메인 페이지 내용
st.title("Welcome to 과제 도우미")
st.write("""
이 애플리케이션은 여러 페이지로 구성되어 있습니다. 
사이드바를 이용해 다른 페이지로 이동하세요.
""")

# 사이드바
st.sidebar.title("Navigation")
st.sidebar.write("사이드바를 통해 페이지를 탐색할 수 있습니다.")
