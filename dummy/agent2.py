import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class IdeaGenerationAgent:
    def __init__(self):
        self.situation_prompts = {
            "새로운 제품 아이디어": {
                "prompt": "새로운 제품 아이디어를 생성해주세요.",
                "considerations": ["시장 수요", "경쟁 제품", "기술적 가능성"]
            },
            "마케팅 전략": {
                "prompt": "효과적인 마케팅 전략을 제시해주세요.",
                "considerations": ["타겟 고객", "광고 채널", "예산"]
            }
        }
        
        # 사용자 정의 프롬프트 템플릿 추가
        self.custom_prompt_templates = {
            "창의성 강화": "더 창의적이고 혁신적인 아이디어를 생성해주세요.",
            "실용성 강화": "더 현실적이고 즉시 실행 가능한 아이디어를 제시해주세요.",
            "비용 효율성": "비용 대비 효과가 높은 방안을 중심으로 제시해주세요.",
            "빠른 실행": "단기간에 실행할 수 있는 방안을 중심으로 제시해주세요.",
            "위험 최소화": "리스크를 최소화할 수 있는 방안을 중심으로 제시해주세요.",
            "확장성": "향후 확장 가능성이 높은 방안을 중심으로 제시해주세요.",
            "사용자 정의": "직접 프롬프트를 입력하세요..."
        }

    def analyze_and_generate(self, text: str, situation: str, custom_prompt: str = "") -> Dict[str, str]:
        """선택된 상황과 추가 프롬프트에 따라 아이디어를 생성"""
        prompt_data = self.situation_prompts[situation]
        base_prompt = f"""
{prompt_data['prompt']}

고려사항:
{', '.join(prompt_data['considerations'])}

추가 요구사항:
{custom_prompt}

응답은 다음 형식으로 구조화해주세요:
1. 상황 분석
2. 핵심 아이디어
3. 실행 방안
4. 기대 효과
5. 고려사항
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": f"다음 내용을 {situation} 관점에서 분석하고 아이디어를 제시해주세요:\n\n{text}"}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"아이디어 생성 중 오류 발생: {str(e)}"}

    def generate_variations(self, idea: str) -> List[str]:
        """아이디어의 변형을 생성"""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "주어진 아이디어의 새로운 변형을 생성해주세요."},
                    {"role": "user", "content": f"다음 아이디어의 변형을 3가지 제시해주세요:\n\n{idea}"}
                ],
                temperature=0.9,
                max_tokens=500
            )
            return self._extract_variations(response.choices[0].message.content)
        except Exception as e:
            return [f"변형 생성 중 오류 발생: {str(e)}"]

    def _parse_response(self, response: str) -> Dict[str, str]:
        sections = response.split("\n\n")
        result = {}
        for section in sections:
            if section.strip():
                title, content = section.split("\n", 1)
                result[title.strip()] = content.strip()
        return result

    def _extract_variations(self, response: str) -> List[str]:
        return [variation.strip() for variation in response.split("\n") if variation.strip()]

def main():
    st.title("💡 상황별 AI 아이디어 생성기")
    
    if 'idea_agent' not in st.session_state:
        st.session_state.idea_agent = IdeaGenerationAgent()

    # 사이드바 구성
    with st.sidebar:
        st.subheader("🎯 상황 선택")
        selected_situation = st.selectbox(
            "아이디어가 필요한 상황을 선택하세요:",
            options=list(st.session_state.idea_agent.situation_prompts.keys())
        )
        
        # 선택된 상황의 고려사항 표시
        st.markdown("---")
        st.subheader("🔍 주요 고려사항")
        for consideration in st.session_state.idea_agent.situation_prompts[selected_situation]["considerations"]:
            st.write(f"- {consideration}")

    # 메인 영역
    st.subheader("1️⃣ 상황 설명")
    input_text = st.text_area(
        "현재 상황이나 고민을 설명해주세요:",
        height=150,
        help=f"선택하신 '{selected_situation}' 상황에 대해 자세히 설명해주세요."
    )

    # 추가 프롬프트 설정
    st.subheader("2️⃣ 추가 요구사항 설정")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        prompt_template = st.selectbox(
            "프롬프트 템플릿 선택:",
            options=list(st.session_state.idea_agent.custom_prompt_templates.keys()),
            key="prompt_template"
        )

    with col2:
        if prompt_template == "사용자 정의":
            custom_prompt = st.text_input(
                "추가 프롬프트를 직접 입력하세요:",
                help="AI에게 주고 싶은 추가적인 지시사항을 입력하세요."
            )
        else:
            custom_prompt = st.session_state.idea_agent.custom_prompt_templates[prompt_template]
            st.text_input("선택된 프롬프트:", value=custom_prompt, disabled=True)

    # 프롬프트 미리보기
    with st.expander("🔍 최종 프롬프트 미리보기", expanded=False):
        st.markdown(f"""
        **선택된 상황**: {selected_situation}
        
        **추가 요구사항**: {custom_prompt}
        
        **주요 고려사항**:
        {', '.join(st.session_state.idea_agent.situation_prompts[selected_situation]["considerations"])}
        """)

    # 실행 버튼
    col3, col4 = st.columns([1, 1])
    
    with col3:
        analyze_button = st.button("아이디어 생성", type="primary")
    with col4:
        if st.button("초기화"):
            st.session_state.clear()
            st.experimental_rerun()
    
    if analyze_button and input_text:
        with st.spinner(f"'{selected_situation}' 상황에 대한 아이디어를 생성하고 있습니다..."):
            analysis_result = st.session_state.idea_agent.analyze_and_generate(
                input_text, 
                selected_situation,
                custom_prompt
            )
            
            if "error" in analysis_result:
                st.error(analysis_result["error"])
            else:
                # 결과 표시
                st.subheader("3️⃣ 생성된 아이디어")
                for section, content in analysis_result.items():
                    with st.expander(section, expanded=True):
                        st.markdown(content)

    # 도움말
    with st.sidebar:
        st.markdown("---")
        st.subheader("💡 효과적인 사용 팁")
        st.markdown("""
        1. **상황 선택**
           - 가장 적합한 상황 유형 선택
           - 관련 고려사항 확인
        
        2. **상황 설명**
           - 구체적인 목표와 제약사항 포함
           - 현재 상황을 상세히 설명
        
        3. **추가 요구사항**
           - 템플릿 활용 또는 직접 입력
           - 원하는 방향성 명확히 제시
        
        4. **결과 활용**
           - 생성된 아이디어를 기반으로 구체화
           - 필요시 다른 관점으로 재시도
        """)

if __name__ == "__main__":
    main()