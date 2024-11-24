import streamlit as st
import openai
import time
from typing import List, Dict
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

class Agent:
    def __init__(self, name: str, personality: str, role: str):
        self.name = name
        self.personality = personality
        self.role = role
        
    def generate_response(self, topic: str, context: List[Dict] = None) -> str:
        if context is None:
            context = []
            
        system_message = f"""
당신은 {self.name}입니다. {self.personality}의 성격을 가지고 있으며, {self.role}의 관점에서 의견을 제시합니다.
다음 규칙을 따라주세요:
1. 주어진 주제에 대해 {self.role}의 관점에서 분석하고 의견을 제시하세요
2. 응답은 2-3문장으로 간단명료하게 해주세요
3. 전문적이면서도 이해하기 쉬운 언어를 사용하세요
"""

        messages = [{"role": "system", "content": system_message}]
        
        for msg in context:
            messages.append({"role": "assistant", "content": msg["content"]})
            
        messages.append({
            "role": "user", 
            "content": f"다음 주제에 대해 {self.role}의 관점에서 의견을 제시해주세요: {topic}"
        })

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API 오류 발생: {str(e)}"

def generate_conclusion(messages: List[Dict]) -> str:
    system_message = """
두 에이전트의 의견을 종합하여 결론을 도출해주세요.
결론은 다음 형식을 따라주세요:
1. 두 의견의 공통점과 차이점을 파악하세요
2. 균형잡힌 관점에서 통합적인 결론을 제시하세요
3. 2-3문장으로 간단명료하게 작성하세요
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"다음 두 의견에 대한 결론을 도출해주세요:\n\n첫 번째 의견: {messages[0]['content']}\n\n두 번째 의견: {messages[1]['content']}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 오류 발생: {str(e)}"

def check_api_key():
    """API 키 유효성 검사"""
    if not openai.api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return False
    return True

def main():
    st.title("🤖 AI 에이전트 토론 시스템")
    
    # API 키 검증
    if not check_api_key():
        return
    
    # 세션 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 에이전트 초기화
    agent1 = Agent(
        name="분석가 에이전트",
        personality="논리적이고 데이터 중심적인",
        role="데이터 분석가"
    )
    agent2 = Agent(
        name="창의적 에이전트",
        personality="혁신적이고 직관적인",
        role="혁신 전문가"
    )
    
    # 사용자 입력
    topic = st.text_input("토론 주제를 입력하세요:", value="인공지능의 미래")
    
    if st.button("토론 시작"):
        st.session_state.messages = []  # 메시지 초기화
        
        with st.spinner("첫 번째 에이전트 생각 중..."):
            thought1 = agent1.generate_response(topic)
            st.session_state.messages.append({"agent": agent1.name, "content": thought1})
        
        with st.spinner("두 번째 에이전트 생각 중..."):
            thought2 = agent2.generate_response(topic, st.session_state.messages)
            st.session_state.messages.append({"agent": agent2.name, "content": thought2})
        
        with st.spinner("결론 도출 중..."):
            conclusion = generate_conclusion(st.session_state.messages)
            st.session_state.messages.append({"agent": "결론", "content": conclusion})
    
    # 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["agent"]):
            st.write(f"**{msg['agent']}**: {msg['content']}")
            time.sleep(0.5)

if __name__ == "__main__":
    main()