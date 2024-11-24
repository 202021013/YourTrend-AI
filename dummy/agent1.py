import streamlit as st
import openai
from dotenv import load_dotenv
import os
from typing import Dict, List

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

class IdeaGenerationAgent:
    def __init__(self):
        self.system_prompt = """
당신은 창의적인 아이디어 생성 전문가입니다. 주어진 텍스트를 분석하고 다음 단계를 수행해주세요:

1. 텍스트의 핵심 개념과 주요 포인트를 파악합니다.
2. 관련된 새로운 아이디어를 생성합니다.
3. 각 아이디어에 대한 구체적인 근거와 실현 가능성을 제시합니다.
4. 실제 적용 방안과 예상되는 영향을 설명합니다.

응답은 다음 형식으로 구조화해주세요:
- 핵심 개념 요약
- 주요 인사이트
- 새로운 아이디어 및 근거
- 적용 방안 및 기대효과
"""

    def analyze_and_generate(self, text: str) -> Dict[str, str]:
        """텍스트를 분석하고 아이디어를 생성"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"다음 텍스트를 분석하고 아이디어를 생성해주세요:\n\n{text}"}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"아이디어 생성 중 오류 발생: {str(e)}"}

    def brainstorm_variations(self, initial_idea: str) -> List[str]:
        """초기 아이디어에 대한 다양한 변형 생성"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "주어진 아이디어의 다양한 변형과 응용 방안을 제시해주세요."},
                    {"role": "user", "content": f"다음 아이디어에 대한 3가지 다른 변형을 제시해주세요:\n\n{initial_idea}"}
                ],
                temperature=0.9,
                max_tokens=500
            )
            return self._extract_variations(response.choices[0].message.content)
        except Exception as e:
            return [f"변형 아이디어 생성 중 오류 발생: {str(e)}"]

    def _parse_response(self, response: str) -> Dict[str, str]:
        """API 응답을 구조화된 형식으로 파싱"""
        sections = ["핵심 개념 요약", "주요 인사이트", "새로운 아이디어 및 근거", "적용 방안 및 기대효과"]
        result = {}
        
        current_section = ""
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if any(section in line for section in sections):
                if current_section and current_content:
                    result[current_section] = '\n'.join(current_content)
                current_section = line
                current_content = []
            else:
                current_content.append(line)
                
        if current_section and current_content:
            result[current_section] = '\n'.join(current_content)
            
        return result

    def _extract_variations(self, response: str) -> List[str]:
        """변형 아이디어 응답을 리스트로 파싱"""
        variations = []
        current_variation = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                if current_variation:
                    variations.append('\n'.join(current_variation))
                    current_variation = []
            else:
                current_variation.append(line)
                
        if current_variation:
            variations.append('\n'.join(current_variation))
            
        return variations[:3]  # 최대 3개의 변형만 반환

def main():
    st.title("💡 AI 아이디어 생성기")
    
    if 'idea_agent' not in st.session_state:
        st.session_state.idea_agent = IdeaGenerationAgent()
    
    # 입력 텍스트 영역
    input_text = st.text_area(
        "분석할 텍스트를 입력하세요:",
        height=150,
        help="아이디어를 생성할 텍스트를 입력해주세요. 더 자세한 텍스트를 입력할수록 더 구체적인 아이디어가 생성됩니다."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_button = st.button("아이디어 생성", type="primary")
    with col2:
        if st.button("초기화"):
            st.session_state.clear()
            st.experimental_rerun()
    
    if analyze_button and input_text:
        with st.spinner("아이디어를 생성하고 있습니다..."):
            # 아이디어 생성
            analysis_result = st.session_state.idea_agent.analyze_and_generate(input_text)
            
            if "error" in analysis_result:
                st.error(analysis_result["error"])
            else:
                # 결과 표시
                for section, content in analysis_result.items():
                    with st.expander(section, expanded=True):
                        st.markdown(content)
                
                # 추가 아이디어 변형 생성
                if "새로운 아이디어 및 근거" in analysis_result:
                    st.subheader("🌟 추가 아이디어 변형")
                    variations = st.session_state.idea_agent.brainstorm_variations(
                        analysis_result["새로운 아이디어 및 근거"]
                    )
                    
                    for i, variation in enumerate(variations, 1):
                        with st.expander(f"아이디어 변형 {i}", expanded=True):
                            st.markdown(variation)
    
    # 사용 가이드
    with st.sidebar:
        st.subheader("📚 사용 가이드")
        st.markdown("""
        1. 분석할 텍스트를 입력합니다.
        2. '아이디어 생성' 버튼을 클릭합니다.
        3. AI가 다음을 생성합니다:
           - 핵심 개념 요약
           - 주요 인사이트
           - 새로운 아이디어 및 근거
           - 적용 방안 및 기대효과
           - 추가 아이디어 변형
        """)
        
        st.subheader("💡 팁")
        st.markdown("""
        - 더 자세한 텍스트를 입력할수록 더 구체적인 아이디어가 생성됩니다.
        - 특정 분야나 컨텍스트를 명시하면 더 관련성 높은 아이디어가 생성됩니다.
        - 초기화 버튼을 사용하여 새로운 세션을 시작할 수 있습니다.
        """)

if __name__ == "__main__":
    main()