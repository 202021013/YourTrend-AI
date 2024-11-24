import streamlit as st
import openai
import time
from typing import List, Dict

class AIAgent:
    def __init__(self, name: str, role: str, api_key: str):
        self.name = name
        self.role = role
        openai.api_key = api_key
        self.conversation_history: List[Dict] = []
        
    def generate_response(self, topic: str, other_response: str = "") -> str:
        messages = [
            {"role": "system", "content": f"""당신은 {self.name}입니다. {self.role}
            다른 에이전트와 대화하면서 주제에 대해 깊이 있게 분석하고 토론하세요.
            각자의 관점에서 중요한 포인트를 제시하고, 상대방의 의견을 발전시켜 나가야 합니다.
            최종적으로는 혁신적인 아이디어 도출을 위한 기반을 만들어야 합니다."""},
            {"role": "user", "content": f"주제: {topic}"}
        ]
        
        messages.extend(self.conversation_history)
        
        if other_response:
            messages.append({"role": "user", "content": f"다른 에이전트의 의견: {other_response}\n이 의견을 고려하여 당신의 전문성을 바탕으로 의견을 발전시켜주세요."})
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            generated_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            
            return generated_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def generate_innovative_conclusion(topic: str, conversation_history: List[str], api_key: str) -> str:
    # 대화 내용을 줄바꿈으로 구분하여 하나의 문자열로 만듦
    conversation_text = ""
    for idx, msg in enumerate(conversation_history, 1):
        conversation_text += f"발언 {idx}: {msg}\n"
    
    prompt = (
        f"주제: {topic}\n"
        f"대화 내용:\n{conversation_text}\n"
        "위 대화를 바탕으로 혁신적인 아이디어와 실행 방안을 도출해주세요."
    )
    
    messages = [
        {"role": "system", "content": """당신은 혁신적인 아이디어를 도출하는 전문가입니다.
        분석가와 실무자의 대화를 종합하여 다음과 같은 결과물을 도출해주세요:
        1. 대화에서 발견된 핵심 인사이트 (2-3줄)
        2. 이를 바탕으로 한 혁신적인 아이디어 제안 (3-4개)
        3. 실현 가능한 실행 방안 (2-3줄)
        아이디어는 구체적이고 실현 가능하면서도 창의적이어야 합니다."""},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=400,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating conclusion: {str(e)}"

def main():
    st.title("💡 AI 아이디어 발굴 시스템")
    st.markdown("### 분석가와 실무자의 대화를 통한 혁신적인 아이디어 도출")
    
    # API 키 입력
    api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    
    if not api_key:
        st.warning("API 키를 입력해주세요.")
        return
        
    # 세션 상태 초기화
    if 'dialogue_started' not in st.session_state:
        st.session_state.dialogue_started = False
        st.session_state.current_step = 0
        st.session_state.conversation_history = []
    
    # 에이전트 초기화
    analyst = AIAgent(
        "분석가", 
        """당신은 데이터와 트렌드를 깊이 있게 분석하는 전문가입니다.
        시장 동향, 소비자 행동, 기술 트렌드 등을 종합적으로 분석하여 인사이트를 도출합니다.
        항상 데이터에 기반한 객관적인 의견을 제시하되, 미래 가능성도 고려합니다.""",
        api_key
    )
    
    practitioner = AIAgent(
        "실무자", 
        """당신은 현장 경험이 풍부한 실무 전문가입니다.
        실제 적용 가능성, 자원 효율성, 실행 시의 문제점 등을 고려합니다.
        현실적인 제약사항을 고려하되, 혁신적인 해결방안을 선호합니다.""",
        api_key
    )
    
    # 주제 입력
    topic = st.text_input("탐구할 주제나 해결할 문제를 입력하세요:", "미래의 스마트 시티 설계")
    
    if st.button("대화 시작") or st.session_state.dialogue_started:
        st.session_state.dialogue_started = True
        
        for i in range(min(st.session_state.current_step + 1, 3)):
            with st.container():
                st.markdown(f"#### 라운드 {i+1}")
                
                # 분석가의 응답
                response_analyst = analyst.generate_response(
                    topic, 
                    st.session_state.conversation_history[-1] if st.session_state.conversation_history else ""
                )
                st.markdown(f"**📊 분석가**: {response_analyst}")
                st.session_state.conversation_history.append(response_analyst)
                time.sleep(1)
                
                # 실무자의 응답
                response_practitioner = practitioner.generate_response(topic, response_analyst)
                st.markdown(f"**🛠️ 실무자**: {response_practitioner}")
                st.session_state.conversation_history.append(response_practitioner)
                time.sleep(1)
        
        if st.session_state.current_step < 2:
            st.session_state.current_step += 1
        elif st.session_state.current_step == 2:
            st.markdown("---")
            st.markdown("### 🚀 혁신적 아이디어 도출")
            conclusion = generate_innovative_conclusion(topic, st.session_state.conversation_history, api_key)
            
            # 결론을 섹션별로 나누어 표시
            sections = conclusion.split('\n\n')
            for section in sections:
                if '인사이트' in section.lower():
                    st.markdown("#### 🔍 핵심 인사이트")
                    st.write(section.split(':', 1)[1] if ':' in section else section)
                elif '아이디어' in section.lower():
                    st.markdown("#### 💡 혁신적 아이디어")
                    st.write(section.split(':', 1)[1] if ':' in section else section)
                elif '실행' in section.lower():
                    st.markdown("#### ⚡ 실행 방안")
                    st.write(section.split(':', 1)[1] if ':' in section else section)
                else:
                    st.write(section)
    
    # 리셋 버튼
    if st.button("새로운 주제로 시작하기"):
        st.session_state.dialogue_started = False
        st.session_state.current_step = 0
        st.session_state.conversation_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()