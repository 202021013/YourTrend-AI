import streamlit as st
from openai import OpenAI
import time
from typing import List, Dict
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

class AIAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.client = OpenAI()  # API 키는 환경변수에서 자동으로 로드됨
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
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            generated_response = response.choices[0].message.content.strip()
            return generated_response
        
        except Exception as e:
            return f"Error generating response: {str(e)}"

def generate_innovative_conclusion(topic: str, conversation_history: List[str], custom_prompt: str) -> str:
    client = OpenAI()
    
    conversation_text = "\n".join(f"발언 {idx}: {msg}" for idx, msg in enumerate(conversation_history, 1))
    
    # 사용자 정의 프롬프트가 있으면 그것을 사용하고, 없으면 기본 프롬프트 사용
    system_prompt = custom_prompt if custom_prompt else """당신은 혁신적인 아이디어를 도출하는 전문가입니다.
        분석가와 실무자의 대화를 종합하여 다음과 같은 결과물을 도출해주세요:
        1. 대화에서 발견된 핵심 인사이트 (2-3줄)
        2. 이를 바탕으로 한 혁신적인 아이디어 제안 (3-4개)
        3. 실현 가능한 실행 방안 (2-3줄)
        아이디어는 구체적이고 실현 가능하면서도 창의적이어야 합니다."""
    
    prompt = (
        f"주제: {topic}\n"
        f"대화 내용:\n{conversation_text}\n"
        "위 대화를 바탕으로 혁신적인 아이디어와 실행 방안을 도출해주세요."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=800,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating conclusion: {str(e)}"

def create_message_container(role: str, message: str):
    """메시지를 스타일이 적용된 컨테이너에 표시합니다."""
    if role == "분석가":
        st.markdown(
            f"""
            <div style="
                background-color: #E8F4F9;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid #2196F3;
            ">
                <strong>📊 분석가</strong><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  # 실무자
        st.markdown(
            f"""
            <div style="
                background-color: #FFF3E0;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid #FF9800;
            ">
                <strong>🛠️ 실무자</strong><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )

def create_round_separator(round_number: int):
    """라운드 구분선과 헤더를 생성합니다."""
    st.markdown(
        f"""
        <div style="
            margin: 30px 0 20px 0;
            padding: 10px 0;
            border-top: 2px solid #e0e0e0;
        ">
            <h3 style="color: #1976D2; margin: 10px 0;">🔄 라운드 {round_number}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("💡 AI 아이디어 발굴 시스템")
    st.markdown("### 분석가와 실무자의 대화를 통한 혁신적인 아이디어 도출")
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다. .env 파일을 확인해주세요.")
        return
    
    # 세션 상태 초기화
    if 'started' not in st.session_state:
        st.session_state.started = False
        st.session_state.conversation_history = []
        st.session_state.current_round = 0
        st.session_state.is_processing = False
    
    # 에이전트 초기화 (이전과 동일)
    analyst = AIAgent(
        "분석가", 
        """당신은 데이터와 트렌드를 깊이 있게 분석하는 전문가입니다.
        시장 동향, 소비자 행동, 기술 트렌드 등을 종합적으로 분석하여 인사이트를 도출합니다.
        항상 데이터에 기반한 객관적인 의견을 제시하되, 미래 가능성도 고려합니다."""
    )
    
    practitioner = AIAgent(
        "실무자", 
        """당신은 현장 경험이 풍부한 실무 전문가입니다.
        실제 적용 가능성, 자원 효율성, 실행 시의 문제점 등을 고려합니다.
        현실적인 제약사항을 고려하되, 혁신적인 해결방안을 선호합니다."""
    )
    
    # 주제 입력
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("탐구할 주제나 해결할 문제를 입력하세요:", "미래의 스마트 시티 설계")
    
    # 아이디어 생성 방향 설정
    with st.expander("🎯 아이디어 생성 방향 설정", expanded=True):
        custom_prompt = st.text_area(
            "아이디어 생성을 위한 구체적인 방향을 설정하세요:",
            """당신은 혁신적인 아이디어를 도출하는 전문가입니다.
분석가와 실무자의 대화를 종합하여 다음과 같은 결과물을 도출해주세요:
1. 대화에서 발견된 핵심 인사이트 (2-3줄)
2. 이를 바탕으로 한 혁신적인 아이디어 제안 (3-4개)
3. 실현 가능한 실행 방안 (2-3줄)
아이디어는 구체적이고 실현 가능하면서도 창의적이어야 합니다.""",
            height=200
        )
        st.info("💡 프롬프트를 수정하여 원하는 방향의 아이디어를 생성하도록 설정할 수 있습니다.")
    
    # 새로운 주제로 시작하기 버튼
    with col2:
        if st.button("새로운 주제로 시작하기"):
            st.session_state.started = False
            st.session_state.conversation_history = []
            st.session_state.current_round = 0
            st.session_state.is_processing = False
            st.rerun()

    # 대화 시작 버튼
    if not st.session_state.started and st.button("대화 시작", type="primary"):
        st.session_state.started = True
        st.session_state.is_processing = True
        st.rerun()
    
    # 대화 진행 (이전 코드와 동일하게 유지하되, conclusion 생성 시 custom_prompt 전달)
    if st.session_state.started:
        # 진행 상태 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 각 라운드 진행
        while st.session_state.current_round < 3 and st.session_state.is_processing:
            progress_value = st.session_state.current_round / 3
            progress_bar.progress(progress_value)
            
            status_text.text(f"라운드 {st.session_state.current_round + 1} 진행 중...")
            
            # 이전 라운드들의 내용을 먼저 표시
            for past_round in range(st.session_state.current_round):
                create_round_separator(past_round + 1)
                past_analyst = st.session_state.conversation_history[past_round * 2]
                past_practitioner = st.session_state.conversation_history[past_round * 2 + 1]
                create_message_container("분석가", past_analyst)
                create_message_container("실무자", past_practitioner)
            
            # 현재 라운드 표시
            create_round_separator(st.session_state.current_round + 1)
            
            # 분석가의 응답
            with st.spinner('분석가가 응답을 생성 중입니다...'):
                response_analyst = analyst.generate_response(
                    topic, 
                    st.session_state.conversation_history[-1] if st.session_state.conversation_history else ""
                )
                create_message_container("분석가", response_analyst)
                st.session_state.conversation_history.append(response_analyst)
            
            # 실무자의 응답
            with st.spinner('실무자가 응답을 생성 중입니다...'):
                response_practitioner = practitioner.generate_response(topic, response_analyst)
                create_message_container("실무자", response_practitioner)
                st.session_state.conversation_history.append(response_practitioner)
            
            st.session_state.current_round += 1
            time.sleep(1)  # UI 업데이트를 위한 짧은 대기
            
            if st.session_state.current_round < 3:
                st.rerun()
        
        # 모든 라운드가 완료되면 결론 도출
        if st.session_state.current_round == 3:
            progress_bar.progress(1.0)
            status_text.text("모든 라운드가 완료되었습니다. 최종 결론을 도출합니다...")
            
            st.markdown("---")
            st.markdown("### 🚀 혁신적 아이디어 도출")
            
            with st.spinner('최종 결론을 도출하고 있습니다...'):
                conclusion = generate_innovative_conclusion(
                    topic, 
                    st.session_state.conversation_history,
                    custom_prompt
                )
                
                # 결론을 섹션별로 나누어 표시
                sections = conclusion.split('\n\n')
                for section in sections:
                    if '인사이트' in section.lower():
                        st.markdown("#### 🔍 핵심 인사이트")
                        st.info(section.split(':', 1)[1] if ':' in section else section)
                    elif '아이디어' in section.lower():
                        st.markdown("#### 💡 혁신적 아이디어")
                        st.success(section.split(':', 1)[1] if ':' in section else section)
                    elif '실행' in section.lower():
                        st.markdown("#### ⚡ 실행 방안")
                        st.warning(section.split(':', 1)[1] if ':' in section else section)
                    else:
                        st.write(section)
            
            status_text.text("분석이 완료되었습니다. '새로운 주제로 시작하기' 버튼을 눌러 새로운 주제를 탐구할 수 있습니다.")

if __name__ == "__main__":
    main()