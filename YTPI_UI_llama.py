import streamlit as st
import whisper
import yt_dlp
#from openai import OpenAI
import os
from dotenv import load_dotenv
from youtubesearchpython import VideosSearch
import pandas as pd
from typing import List, Dict
import ollama
load_dotenv()

def clean_view_count(view_data: dict) -> int:
    """조회수 데이터에서 숫자 추출"""
    try:
        if isinstance(view_data, dict):
            view_text = view_data.get('short', '0')
        else:
            view_text = str(view_data)

        number = ''.join(filter(lambda x: x.isdigit() or x in 'KMB.', view_text.upper()))
        
        if not number:
            return 0

        multiplier = 1
        if 'K' in number:
            multiplier = 1000
            number = number.replace('K', '')
        elif 'M' in number:
            multiplier = 1000000
            number = number.replace('M', '')
        elif 'B' in number:
            multiplier = 1000000000
            number = number.replace('B', '')

        return int(float(number) * multiplier)
    except Exception as e:
        return 0

def truncate_to_complete_sentence(text: str, max_tokens: int) -> str:
    estimated_tokens = len(text.split()) * 1.3
    
    if estimated_tokens <= max_tokens:
        return text
        
    approx_chars = int(max_tokens * 4)
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    truncated_text = text[:approx_chars]
    
    last_sentence_end = -1
    for ending in sentence_endings:
        pos = truncated_text.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos
            
    if last_sentence_end != -1:
        return text[:last_sentence_end + 2].strip()
    
    last_space = truncated_text.rfind(' ')
    if last_space != -1:
        return text[:last_space].strip() + "..."
        
    return truncated_text.strip() + "..."

class AIAgent:
    def __init__(self, name: str, role: str, temperature: float, personality: str):
        self.name = name
        self.role = role
        self.temperature = temperature
        self.personality = personality
        self.conversation_history: List[Dict] = []
        
    def generate_response(self, topic: str, other_response: str = "", context: str = "", round_num: int = 1) -> str:
        if round_num == 1:
            prompt = f"""
당신은 {self.name}이며, {self.role}입니다.
성격과 말투: {self.personality}

토론 주제: {topic}

분석할 콘텐츠:
{context}

다음 형식으로 의견을 제시해주세요:
1. 현재 상황 분석
2. 기회 요소 발견
3. 해결 방안 제시
4. 구체적 실행 계획
5. 예상되는 도전 과제
"""
        else:
            prompt = f"""
당신은 {self.name}이며, {self.role}입니다.
성격과 말투: {self.personality}

이전 대화:
{other_response}

위 내용에 대한 짧은 피드백과 제안을 200자 이내로 제시해주세요.
"""
    
        try:
            response = ollama.chat(
                model="llama3.2:latest",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": topic}
                ],
                stream=False,
                options={
                    "temperature": self.temperature,
                    "num_predict": 200 if round_num > 1 else 1000,
                }
            )
            
            generated_response = response.message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            
            return generated_response

        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"
        
        
def search_videos(keyword: str, duration: str = 'any', sort: str = 'relevance') -> pd.DataFrame:
    try:
        videos_search = VideosSearch(keyword, limit=10)
        search_result = videos_search.result()
        
        if not search_result or 'result' not in search_result:
            return pd.DataFrame()
            
        results = []
        
        for video in search_result['result']:
            try:
                duration_str = video.get('duration', '0:00')
                duration_parts = duration_str.split(':')
                total_minutes = 0
                
                if len(duration_parts) == 2:  # MM:SS
                    total_minutes = int(duration_parts[0])
                elif len(duration_parts) == 3:  # HH:MM:SS
                    total_minutes = int(duration_parts[0]) * 60 + int(duration_parts[1])
                
                if duration == 'short' and total_minutes > 5:
                    continue
                elif duration == 'medium' and (total_minutes <= 5 or total_minutes > 15):
                    continue
                elif duration == 'long' and total_minutes <= 15:
                    continue
                
                view_count = clean_view_count(video.get('viewCount', {}))
                thumbnails = video.get('thumbnails', [])
                thumbnail_url = thumbnails[0].get('url', '') if thumbnails else ''
                
                results.append({
                    'video_id': video.get('id', ''),
                    'title': video.get('title', '').strip(),
                    'url': f"https://www.youtube.com/watch?v={video.get('id', '')}",
                    'thumbnail': thumbnail_url,
                    'duration': duration_str,
                    'view_count': view_count,
                    'author': video.get('channel', {}).get('name', '').strip()
                })
                
            except Exception as e:
                st.warning(f"비디오 정보 처리 중 오류 발생: {str(e)}")
                continue
        
        if not results:
            st.warning("검색 결과가 없습니다.")
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        if sort == 'date':
            if 'publishedTime' in df.columns:
                df = df.sort_values('publishedTime', ascending=False)
        elif sort == 'views':
            df = df.sort_values('view_count', ascending=False)
            
        return df
        
    except Exception as e:
        st.error(f"영상 검색 중 오류 발생: {str(e)}")
        return pd.DataFrame()

def format_views(view_count: int) -> str:
    try:
        if not isinstance(view_count, (int, float)):
            return "0"
            
        if view_count >= 1000000000:
            return f"{view_count/1000000000:.1f}B"
        elif view_count >= 1000000:
            return f"{view_count/1000000:.1f}M"
        elif view_count >= 1000:
            return f"{view_count/1000:.1f}K"
        return str(view_count)
    except:
        return "0"

def download_audio(video_url: str) -> str:
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio_%(id)s.%(ext)s'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = f"audio_{info['id']}.mp3"
            return audio_path
            
    except Exception as e:
        st.error(f"오디오 다운로드 중 오류 발생: {str(e)}")
        return None

# def transcribe_audio(audio_path: str) -> str:
#     try:
#         # CUDA GPU 사용 가능 여부 확인
#         import torch
#         device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         if device == "cpu":
#             st.warning("⚠️ GPU가 감지되지 않아 CPU에서 실행됩니다. 음성 인식 속도가 느릴 수 있습니다.")
#         else:
#             st.info("🚀 GPU를 사용하여 음성을 인식합니다.")
        
#         # Whisper 모델 로드 시 device 지정
#         model = whisper.load_model("medium", device=device)
        
#         # GPU 메모리 최적화를 위한 설정
#         if device == "cuda":
#             model.to(device)
        
#         # transcribe 시 device 지정
#         result = model.transcribe(
#             audio_path,
#             fp16=False if device == "cpu" else True  # GPU일 때만 FP16 사용
#         )
        
#         return result["text"]
        
#     except Exception as e:
#         st.error(f"음성 인식 중 오류 발생: {str(e)}")
#         return None

def transcribe_audio(audio_path: str) -> str:
    try:
        import platform
        import torch

        # 운영 체제 확인
        system = platform.system()

        # 디바이스 설정
        if system == "Darwin" and torch.backends.mps.is_available():
            device = "mps"
            st.info("🚀 macOS에서 M2 칩 GPU(MPS)를 사용하여 음성을 인식합니다.")
        elif system == "Windows" and torch.cuda.is_available():
            device = "cuda"
            st.info("🚀 Windows에서 CUDA GPU를 사용하여 음성을 인식합니다.")
        else:
            device = "cpu"
            if system == "Darwin":
                st.warning("⚠️ macOS에서 GPU(MPS)가 감지되지 않아 CPU에서 실행됩니다.")
            elif system == "Windows":
                st.warning("⚠️ Windows에서 CUDA GPU가 감지되지 않아 CPU에서 실행됩니다.")
            else:
                st.info("💻 지원되지 않는 운영 체제에서 CPU를 사용합니다.")

        # Whisper 모델 로드
        model = whisper.load_model("medium", device=device)

        # transcribe 실행
        result = model.transcribe(
            audio_path,
            fp16=False if device in ["cpu", "mps"] else True  # FP16 설정
        )

        return result["text"]
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        return ""
def create_context(transcripts: List[str], video_urls: List[str]) -> str:
    return "".join([
        f"\n[영상 {i+1}] URL: {url}\n영상 내용 요약:\n{transcript}\n{'-'*50}"
        for i, (transcript, url) in enumerate(zip(transcripts, video_urls))
    ])

def display_message(agent_name: str, message: str):
    style = {
        "시장분석가": {
            "bg_color": "#E8F4F9",
            "border_color": "#2196F3",
            "icon": "📊"
        },
        "프로덕트 매니저": {
            "bg_color": "#F3E5F5",
            "border_color": "#9C27B0",
            "icon": "💡"
        },
        "테크리드": {
            "bg_color": "#E8F5E9",
            "border_color": "#4CAF50",
            "icon": "⚙️"
        },
        "사업전략가": {
            "bg_color": "#FFF3E0",
            "border_color": "#FF9800",
            "icon": "📈"
        }
    }
    
    agent_style = style.get(agent_name, {
        "bg_color": "#F5F5F5",
        "border_color": "#9E9E9E",
        "icon": "💭"
    })
    
    st.markdown(f"""
        <div style="
            background-color: {agent_style['bg_color']};
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid {agent_style['border_color']};
        ">
            <strong>{agent_style['icon']} {agent_name}</strong><br>{message}
        </div>
    """, unsafe_allow_html=True)

def generate_discussion(transcripts: List[str], video_urls: List[str], user_prompt: str) -> tuple:
    analyst = AIAgent(
        name="시장분석가",
        role="시장 트렌드와 사용자 니즈 분석 전문가",
        temperature=0.7,
        personality="데이터 기반의 객관적인 분석을 제공하며, 시장의 기회와 위험 요소를 파악합니다."
    )
    
    product_manager = AIAgent(
        name="프로덕트 매니저",
        role="제품 기획 및 전략 수립 전문가",
        temperature=0.8,
        personality="사용자 중심적 사고와 비즈니스 가치를 균형있게 고려합니다."
    )
    
    tech_lead = AIAgent(
        name="테크리드",
        role="기술 구현 및 아키텍처 설계 전문가",
        temperature=0.7,
        personality="최신 기술 트렌드를 이해하고 실제 구현 가능성을 평가합니다."
    )
    
    business_strategist = AIAgent(
        name="사업전략가",
        role="비즈니스 모델 및 수익화 전략 전문가",
        temperature=0.8,
        personality="시장성과 수익성을 고려한 사업 전략을 수립합니다."
    )

    context = create_context(transcripts, video_urls)
    conversation = []
    agents = [analyst, product_manager, tech_lead, business_strategist]
    rounds = 3
    
    for round_num in range(rounds):
        st.markdown(f"### 🔄 라운드 {round_num + 1}")
        
        for agent in agents:
            with st.spinner(f'{agent.name}의 의견을 분석 중...'):
                other_responses = "\n\n".join([
                    f"{msg['agent']}: {msg['response']}"
                    for msg in conversation[-4:] if msg['agent'] != agent.name
                ])
                
                response = agent.generate_response(
                    user_prompt, 
                    other_responses, 
                    context,
                    round_num + 1
                )
                
                conversation.append({
                    "agent": agent.name,
                    "response": response,
                    "round": round_num + 1
                })
                
                display_message(agent.name, response)
        
        st.markdown(f"""
        <div style="padding: 10px; margin: 20px 0; text-align: center; background-color: #f0f2f6; border-radius: 10px;">
            ✨ 라운드 {round_num + 1} 완료
        </div>
        """, unsafe_allow_html=True)
    
    final_summary = generate_final_summary(conversation, user_prompt)
    return final_summary, conversation

def generate_final_summary(conversation: List[dict], user_prompt: str) -> str:
    conversation_summary = "\n\n".join([
        f"라운드 {msg['round']} - {msg['agent']}: {msg['response'][:300]}"
        for msg in conversation
    ])
    
    prompt = f"""
주제: {user_prompt}

전문가들의 논의 내용을 바탕으로 다음 형식으로 핵심적인 제안 사항만 간단히 정리해주세요:

1. 프로젝트 개요
2. 핵심 기능
3. 기술 구현 방안
4. 비즈니스 전략
5. 실행 계획
6. 주요 고려사항

전문가 논의 내용:
{conversation_summary}
"""

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            options={
                "temperature": 0.7,
                "num_predict": 1500,
            }
        )
        
        return response.message.content.strip()
        
    except Exception as e:
        return f"최종 요약 생성 중 오류 발생: {str(e)}"


def generate_idea_from_videos(selected_videos: List[str], user_prompt: str):
    try:
        transcripts = []
        progress_bar = st.progress(0)
        
        # 영상 처리 최적화
        with st.spinner('영상에서 텍스트를 추출하고 있습니다...'):
            for i, video_url in enumerate(selected_videos):
                audio_path = download_audio(video_url)
                if audio_path:
                    transcript = transcribe_audio(audio_path)
                    if transcript:
                        # 트랜스크립트 길이 제한
                        transcripts.append(truncate_to_complete_sentence(transcript, 1000))
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                progress_bar.progress((i + 1) / len(selected_videos))

        if not transcripts:
            st.error("❌ 선택된 영상에서 텍스트를 추출할 수 없습니다.")
            return None, None

        # 트랜스크립트 표시 최적화
        st.markdown("### 📝 추출된 영상 스크립트")
        with st.expander("스크립트 전체 보기", expanded=False):
            for i, (transcript, url) in enumerate(zip(transcripts, selected_videos), 1):
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border: 1px solid #dee2e6;">
                    <strong>영상 {i}</strong>: {url}
                    <hr style="margin: 0.5rem 0;">
                    <div style="white-space: pre-wrap;">{transcript[:500]}...</div>
                </div>
                """, unsafe_allow_html=True)

        # 선택된 모델로 토론 진행
        model = st.session_state.get('selected_model', 'llama-2')
        st.markdown(f"### 🤖 AI 전문가 토론 시작 (모델: {model})")
        
        final_summary, conversation = generate_discussion(
            transcripts, 
            selected_videos, 
            user_prompt
        )
            
        return final_summary, conversation
        
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return None, None

def render_enhanced_sidebar():
    with st.sidebar:
        st.header("🛠️ 프로젝트 설정")
        
        # 진행 상태 표시 (기존 코드 유지)
        st.header("📊 진행 상태")
        current_step = 1
        if 'selected_videos' in st.session_state and st.session_state.selected_videos:
            current_step = 2
        if 'final_summary' in st.session_state and st.session_state.final_summary:
            current_step = 3
            
        progress_bar = st.progress(current_step / 3)
        st.markdown(f"""
        1. 영상 선택 {'✅' if current_step >= 1 else ''}
        2. 전문가 토론 {'✅' if current_step >= 2 else ''}
        3. 최종 제안서 {'✅' if current_step >= 3 else ''}
        """)
        
        st.markdown("---")
        
        # AI 설정
        st.header("🤖 AI 설정")
        temperature = st.slider(
            "AI 창의성 수준",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="높을수록 더 창의적인 결과가 생성됩니다."
        )
        
        max_tokens = st.select_slider(
            "응답 길이",
            options=[1000, 1500, 2000, 2500],
            value=2000,
            format_func=lambda x: f"{x} 토큰",
            help="생성될 응답의 최대 길이를 설정합니다."
        )
        
        st.markdown("---")
        
        # 프로젝트 정보
        if 'selected_videos' in st.session_state and st.session_state.selected_videos:
            st.header("📊 프로젝트 통계")
            st.write(f"선택된 영상: {len(st.session_state.selected_videos)}개")
            
            if 'conversation_history' in st.session_state:
                total_rounds = len(set(msg['round'] for msg in st.session_state.conversation_history))
                st.write(f"진행된 토론 라운드: {total_rounds}회")
        
        st.markdown("---")
        
        # 빠른 액션
        st.header("⚡ 빠른 액션")
        if st.button("프로젝트 초기화", use_container_width=True):
            for key in ['selected_videos', 'final_summary', 'conversation_history', 
                      'search_results', 'search_performed']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if 'final_summary' in st.session_state:
            if st.button("결과 다운로드 (PDF)", use_container_width=True):
                # PDF 다운로드 로직 구현 필요
                st.info("PDF 다운로드 기능은 곧 제공될 예정입니다.")
        
        # 도움말 섹션
        st.markdown("---")
        with st.expander("❓ 도움말"):
            st.markdown("""
            **사용 방법**
            1. 참고할 유튜브 영상을 검색하고 선택하세요
            2. 프로젝트 요구사항을 상세히 작성하세요
            3. AI 전문가 토론을 시작하세요
            
            **팁**
            - 3개 이상의 영상을 선택하면 더 다양한 인사이트를 얻을 수 있습니다
            - AI 창의성 수준을 조절하여 다양한 결과를 얻어보세요
            """)

def main():
    st.set_page_config(
        page_title="유튜브 프로젝트 아이디어 생성기",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 세션 상태 초기화 (기존 코드 유지)
    if 'selected_videos' not in st.session_state:
        st.session_state.selected_videos = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'discussion_rounds' not in st.session_state:
        st.session_state.discussion_rounds = 2
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'llama-2'
        
    # 향상된 사이드바 렌더링
    render_enhanced_sidebar()
    
    # 기존의 메인 컨테이너 코드는 그대로 유지
    # ... (나머지 코드)

    # CSS 스타일 정의
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #FF0000;
            color: white;
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            border: none;
            min-height: 0px;
            height: auto;
            line-height: 1.5;
            font-size: 0.85rem;
            width: auto !important;
            display: inline-block;
        }
        .stButton>button:hover {
            background-color: #CC0000;
        }
        .video-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        .expert-opinion {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #1a73e8;
        }
        .discussion-round {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 1.5rem 0;
        }
        .final-summary {
            background-color: #e8f0fe;
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
            border: 1px solid #4285f4;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎥 유튜브 프로젝트 아이디어 생성기")
    st.markdown("##### AI 전문가들의 토론을 통해 혁신적인 프로젝트 아이디어를 발굴하세요")

    # 사이드바 구성
    with st.sidebar:
        st.header("프로젝트 진행 단계")
        current_step = 1
        if 'selected_videos' in st.session_state and st.session_state.selected_videos:
            current_step = 2
        if 'final_summary' in st.session_state and st.session_state.final_summary:
            current_step = 3
            
        progress_bar = st.progress(current_step / 3)
        st.markdown(f"""
        1. 영상 선택 {'✅' if current_step >= 1 else ''}
        2. 전문가 토론 {'✅' if current_step >= 2 else ''}
        3. 최종 제안서 {'✅' if current_step >= 3 else ''}
        """)

    # 메인 컨테이너
    with st.container():
        st.header("🔍 참고할 유튜브 영상 검색")
        with st.form(key='search_form'):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                search_keyword = st.text_input(
                    "검색어 입력",
                    placeholder="분석하고 싶은 주제나 키워드를 입력하세요..."
                )
            with col2:
                duration_option = st.selectbox(
                    "영상 길이",
                    options=['any', 'short', 'medium', 'long'],
                    format_func=lambda x: {
                        'any': '전체',
                        'short': '5분 이하',
                        'medium': '5-15분',
                        'long': '15분 이상'
                    }[x]
                )
            with col3:
                sort_option = st.selectbox(
                    "정렬 기준",
                    options=['relevance', 'date', 'views'],
                    format_func=lambda x: {
                        'relevance': '관련도순',
                        'date': '최신순',
                        'views': '조회수순'
                    }[x]
                )
            
            search_submitted = st.form_submit_button("검색", use_container_width=True)

        if search_submitted and search_keyword:
            with st.spinner('🔍 영상을 검색하고 있습니다...'):
                videos_df = search_videos(search_keyword, duration_option, sort_option)
                if not videos_df.empty:
                    st.session_state.search_results = videos_df
                    st.session_state.search_performed = True

        # 검색 결과 표시
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results is not None:
            for _, video in st.session_state.search_results.iterrows():
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"""
                    <div class="video-card">
                        <div style="display: flex; align-items: start;">
                            <img src="{video['thumbnail']}" style="width: 200px; border-radius: 10px;"/>
                            <div style="margin-left: 20px; flex-grow: 1;">
                                <h3>{video['title']}</h3>
                                <p>👤 {video['author']}</p>
                                <p>⏱️ {video['duration']} | 👁️ {format_views(video['view_count'])}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    if video['url'] not in st.session_state.selected_videos:
                        if st.button('선택', key=f"select_{video['video_id']}"):
                            st.session_state.selected_videos.append(video['url'])
                            st.success("✅ 영상이 추가되었습니다!")
                            st.rerun()
                    else:
                        st.warning("⚠️ 선택됨")

        # 선택된 영상 목록
        if st.session_state.selected_videos:
            st.markdown("---")
            st.header("📌 선택된 영상 목록")
            
            for idx, url in enumerate(st.session_state.selected_videos):
                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(f"""
                    <div class="video-card">
                        {idx + 1}. {url}
                    </div>
                    """, unsafe_allow_html=True)
                with cols[1]:
                    if st.button('제거', key=f'remove_{idx}'):
                        st.session_state.selected_videos.pop(idx)
                        st.rerun()

            # 프로젝트 요구사항 입력
            st.markdown("---")
            st.header("💡 프로젝트 요구사항 설정")
            
            user_prompt = st.text_area(
                "프로젝트 요구사항 설명",
                placeholder="어떤 프로젝트를 만들고 싶으신가요? 목표와 주요 요구사항을 자세히 설명해주세요.",
                height=150,
                key="project_requirements"
            )

            if st.button('AI 전문가 토론 시작하기', use_container_width=True):
                if not user_prompt.strip():
                    st.warning("⚠️ 프로젝트 요구사항을 입력해주세요.")
                else:
                    with st.spinner('전문가 토론을 시작합니다...'):
                        final_summary, conversation_history = generate_idea_from_videos(
                            st.session_state.selected_videos,
                            user_prompt
                        )
                        
                        if final_summary and conversation_history:
                            st.session_state.final_summary = final_summary
                            st.session_state.conversation_history = conversation_history

        # 최종 결과 표시
        if 'final_summary' in st.session_state and st.session_state.final_summary:
            st.markdown("---")
            st.header("✨ 최종 프로젝트 제안서")
            st.markdown(f"""
            <div class="final-summary">
                {st.session_state.final_summary}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button('새 프로젝트 시작하기', key='new_project', use_container_width=True):
                for key in ['selected_videos', 'final_summary', 'conversation_history', 
                          'search_results', 'search_performed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()