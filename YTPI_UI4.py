import streamlit as st
import whisper
import yt_dlp
from openai import OpenAI
import sqlite3
import os
from dotenv import load_dotenv
from youtubesearchpython import VideosSearch
import pandas as pd
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# OpenAI 클라이언트 초기화
load_dotenv()

# AI 에이전트 클래스 정의
class AIAgent:
    def __init__(self, name: str, role: str, temperature: float):
        self.name = name
        self.role = role
        self.temperature = temperature
        self.client = OpenAI()
        self.conversation_history: List[Dict] = []
        
    def generate_response(self, topic: str, other_response: str = "") -> str:
        messages = [
            {"role": "system", "content": f"{self.name}: {self.role}"},
            {"role": "user", "content": f"{topic[:3000]}"}
        ]
        
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-4:]
            
        messages.extend(self.conversation_history)
        
        if other_response:
            messages.append({
                "role": "user", 
                "content": f"이전 의견을 고려한 새로운 분석: {other_response[:1000]}"
            })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=800,
                temperature=self.temperature
            )
            
            generated_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            
            return generated_response
            
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {str(e)}")
            return f"Error generating response: {str(e)}"

def summarize_transcript(transcript: str, max_chars: int = 2000) -> str:
    """긴 스크립트를 요약하여 반환합니다."""
    if len(transcript) <= max_chars:
        return transcript
        
    sentences = transcript.split('.')
    total_sentences = len(sentences)
    first_part = sentences[:total_sentences//4]
    last_part = sentences[-total_sentences//4:]
    
    summarized = '. '.join(first_part + ['...'] + last_part)
    return summarized.strip()

def generate_idea(transcripts, video_urls, user_prompt):
    try:
        summarized_transcripts = [summarize_transcript(transcript) for transcript in transcripts]
        
        combined_transcript = "".join([
            f"\n[영상 {i}] URL: {url}\n===== 요약된 스크립트 =====\n{transcript}\n{'-' * 30}\n"
            for i, (transcript, url) in enumerate(zip(summarized_transcripts, video_urls), 1)
        ])

        context = f"""
주요 요구사항:
{user_prompt}

분석할 콘텐츠 요약:
{combined_transcript}
"""

        analyst = AIAgent(
            "분석가", 
            f"""YouTube 콘텐츠 분석 전문가로서, 다음 요구사항에 초점을 맞춰 분석해주세요:
            {user_prompt}""",
            temperature=0.3
        )
        
        practitioner = AIAgent(
            "실무자", 
            f"""프로젝트 실무 전문가로서, 다음 요구사항을 충족하는 실현 가능한 아이디어를 제시해주세요:
            {user_prompt}""",
            temperature=0.7
        )

        conversation_history = []
        
        for round_num in range(3):
            create_round_separator(round_num + 1)
            
            with st.spinner('분석가가 응답�� 생성 중입니다...'):
                response_analyst = analyst.generate_response(
                    context[:3000],  
                    conversation_history[-1][:1000] if conversation_history else ""
                )
                create_message_container("분석가", response_analyst)
                conversation_history.append(response_analyst)
            
            with st.spinner('실무자가 응답을 생성 중입니다...'):
                response_practitioner = practitioner.generate_response(
                    context[:3000],  
                    response_analyst[:1000]  
                )
                create_message_container("실무자", response_practitioner)
                conversation_history.append(response_practitioner)

        summarized_conversation = '. '.join([
            conv[:500] for conv in conversation_history
        ])

        custom_prompt = f"""YouTube 콘텐츠 기반 프로젝트 아이디어 전문가로서,
다음 요구사항에 맞춰 아이디어를 도출해주세요:
{user_prompt}

형식:
1. 프로젝트 개요
2. 세부 구현 방안
3. 차별화 포인트
4. 수익화 전략
5. 개발 로드맵"""

        with st.spinner('최종 아이디어를 생성하고 있습니다...'):
            final_idea = generate_innovative_conclusion(
                user_prompt,
                [summarized_conversation],
                custom_prompt
            )
        
        return final_idea, combined_transcript

    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return "", ""

def generate_innovative_conclusion(topic: str, conversation_history: List[str], custom_prompt: str) -> str:
    client = OpenAI()
    
    conversation_text = "\n".join([
        f"요약 {idx}: {msg[:500]}" 
        for idx, msg in enumerate(conversation_history, 1)
    ])
    
    messages = [
        {"role": "system", "content": custom_prompt[:1000]},
        {"role": "user", "content": f"주제: {topic}\n요약된 대화:\n{conversation_text[:2000]}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1200,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"결론 생성 중 오류 발생: {str(e)}")
        return f"Error generating conclusion: {str(e)}"

# 유틸리티 함수들
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
    else:
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

st.set_page_config(
    page_title="유튜브 프로젝트 아이디어 생성기",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        padding: 0.25rem 0.75rem;  /* 패딩 축소 */
        border: none;
        min-height: 0px;
        height: auto;
        line-height: 1.5;
        font-size: 0.85rem;  /* 글자 크기 축소 */
        width: auto !important;  /* 강제로 자동 너비 적용 */
        display: inline-block;  /* 인라인 블록으로 변경 */
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
    .button-container {
        text-align: right;  /* 버튼 우측 정렬 */
        margin-top: 5px;
    }
    .selected-video {
        background-color: #e9ecef;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    h1 {
        color: #FF0000;
        font-weight: 700;
    }
    h2 {
        color: #1a1a1a;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def get_korean_trends():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
    }
    
    url = "https://trends.google.co.kr/trends/trendingsearches/daily/rss?geo=KR"
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'xml')
        
        trends = []
        items = soup.find_all('item')
        
        for i, item in enumerate(items[:10], 1):
            title = item.find('title').text
            traffic = item.find('ht:approx_traffic').text
            trends.append({"rank": i, "title": title, "traffic": traffic})
            
        return trends
    except Exception as e:
        return [{"rank": 1, "title": f"Error: {e}", "traffic": "N/A"}]
# 영상 스크립트 및 GPT 부분 수정
# .env 파일 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS ideas')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_urls TEXT,
            transcript TEXT,
            idea TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def safe_int_conversion(value, default=0):
    if value is None:
        return default
    try:
        value = str(value).upper()
        if 'K' in value:
            return int(float(value.replace('K', '')) * 1000)
        elif 'M' in value:
            return int(float(value.replace('M', '')) * 1000000)
        return int(float(value.replace(',', '').replace(' views', '')))
    except (ValueError, TypeError, AttributeError):
        return default

def parse_duration(duration_text):
    if not duration_text:
        return 0
    try:
        parts = duration_text.split(':')
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 1:
            return parts[0]
        else:
            return 0
    except:
        return 0

def search_videos(keyword, duration_category, sort_by='relevance'):
    videos_search = VideosSearch(keyword, limit=10)
    results = videos_search.result()
    videos = []
    for video in results['result']:
        video_duration_sec = parse_duration(video['duration'])
        duration_included = False
        if duration_category == 'any':
            duration_included = True
        elif duration_category == 'very_short' and video_duration_sec <= 60:
            duration_included = True
        elif duration_category == 'short' and 60 < video_duration_sec <= 300:
            duration_included = True
        elif duration_category == 'medium' and 300 < video_duration_sec <= 900:
            duration_included = True
        elif duration_category == 'medium_long' and 900 < video_duration_sec <= 1800:
            duration_included = True
        elif duration_category == 'long' and 1800 < video_duration_sec <= 3600:
            duration_included = True
        elif duration_category == 'very_long' and video_duration_sec > 3600:
            duration_included = True
        if duration_included:
            view_count = safe_int_conversion(video['viewCount']['text'])
            publish_time = video.get('publishedTime', '')
            videos.append({
                'video_id': video['id'],
                'title': video['title'],
                'author': video['channel']['name'],
                'duration': video['duration'],
                'duration_seconds': video_duration_sec,
                'view_count': video['viewCount']['text'],
                'view_count_num': view_count,
                'publish_time': publish_time,
                'thumbnail': video['thumbnails'][0]['url'],
                'url': video['link']
            })
    df = pd.DataFrame(videos)
    if sort_by == 'views':
        df = df.sort_values('view_count_num', ascending=False)
    elif sort_by == 'date':
        df['publish_datetime'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df = df.sort_values('publish_datetime', ascending=False)
        df = df.drop('publish_datetime', axis=1)
    return df

def download_audio(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("유효하지 않은 YouTube URL입니다.")
            return None
        
        output_path = f'temp_audio_{video_id}'
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{output_path}.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f'{output_path}.mp3'
    except Exception as e:
        st.error(f"오디오 다운로드 중 오류 발생: {str(e)}")
        return None

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"음성 변환 중 오류 발생: {str(e)}")
        return ""

# 기존 코드의 generate_idea 함수를 AI 에이전트 시스템으로 대체
def generate_idea_from_videos(selected_videos, user_prompt):
    """선택된 영상들로부터 아이디어를 생성하는 통합 함수"""
    try:
        transcripts = []
        progress_bar = st.progress(0)
        
        # 영상 처리 및 텍스트 추출
        for i, video_url in enumerate(selected_videos):
            audio_path = download_audio(video_url)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                if transcript:
                    transcripts.append(transcript)
                try:
                    os.remove(audio_path)
                except:
                    pass
            progress_bar.progress((i + 1) / len(selected_videos))
        
        if not transcripts:
            st.error("❌ 선택된 영상에서 텍스트를 추출할 수 없습니다.")
            return None, None

        # 추출된 스크립트 표시
        st.markdown("### 📝 추출된 영상 스크립트")
        with st.expander("스크립트 전체 보기", expanded=False):
            for i, (transcript, url) in enumerate(zip(transcripts, selected_videos), 1):
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border: 1px solid #dee2e6;">
                    <strong>영상 {i}</strong>: {url}
                    <hr style="margin: 0.5rem 0;">
                    <div style="white-space: pre-wrap;">{transcript}</div>
                </div>
                """, unsafe_allow_html=True)

        # AI 에이전트를 통한 아이디어 생성
        st.markdown("### 🤖 AI 에이전트 분석 진행")
        idea, combined_transcript = generate_idea(transcripts, selected_videos, user_prompt)
            
        return combined_transcript, idea
        
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return None, None

def save_idea(video_urls, transcript, idea):
    try:
        conn = sqlite3.connect('project_ideas.db')
        c = conn.cursor()
        urls_str = ", ".join(video_urls)
        c.execute('INSERT INTO ideas (video_urls, transcript, idea) VALUES (?, ?, ?)', (urls_str, transcript, idea))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"아이디어 저장 중 오류 발생: {str(e)}")

def format_views(view_count):
    if not view_count:
        return "조회수 정보 없음"
    try:
        cleaned_count = view_count.lower().replace('views', '').replace(',', '').strip()
        if 'k' in cleaned_count.lower():
            num = float(cleaned_count.lower().replace('k', ''))
            return f"{num}K"
        elif 'm' in cleaned_count.lower():
            num = float(cleaned_count.lower().replace('m', ''))
            return f"{num}M"
        else:
            count = float(cleaned_count)
            if count >= 1000000:
                return f"{count/1000000:.1f}M"
            elif count >= 1000:
                return f"{count/1000:.1f}K"
            else:
                return str(int(count))
    except (ValueError, TypeError, AttributeError):
        return "조회수 정보 없음"

def init_session_state():
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_videos' not in st.session_state:
        st.session_state.selected_videos = []
    if 'generated_idea' not in st.session_state:
        st.session_state.generated_idea = None
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

def main():
    init_session_state()

    if st.button('새 프로젝트 시작하기', key='new_project_button_top', use_container_width=True):
        st.session_state.generated_idea = None
        st.session_state.selected_videos = []
        st.session_state.search_results = None
        st.session_state.search_performed = False
        st.rerun()

    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #FF0000;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 1rem;
            border: none;
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
        .selected-video {
            background-color: #e9ecef;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        h1 {
            color: #FF0000;
            font-weight: 700;
        }
        h2 {
            color: #1a1a1a;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
# Google Trends 섹션 추가
    st.header("📈 실시간 구글 트렌드")
    with st.spinner("트렌드 정보를 불러오는 중..."):
        trends = get_korean_trends()
        
    cols = st.columns(2)
    for i in range(0, len(trends), 2):
        with cols[0]:
            if i < len(trends):
                trend = trends[i]
                st.markdown(f"""
                <div class="trend-card">
                    <strong>#{trend['rank']}</strong> {trend['title']}
                    <br><small>검색량: {trend['traffic']}</small>
                </div>
                """, unsafe_allow_html=True)
                
        with cols[1]:
            if i + 1 < len(trends):
                trend = trends[i + 1]
                st.markdown(f"""
                <div class="trend-card">
                    <strong>#{trend['rank']}</strong> {trend['title']}
                    <br><small>검색량: {trend['traffic']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.title("🎥 유튜브 프로젝트 아이디어 생성기")
    st.markdown("##### AI를 활용하여 유튜브 영상에서 혁신적인 프로젝트 아이디어를 발굴하세요")
    init_db()

    with st.sidebar:
        st.header("진행 상황")
        total_steps = 3
        current_step = 1 if not st.session_state.selected_videos else (
            3 if st.session_state.generated_idea else 2)
        st.progress(current_step / total_steps)
        
        st.markdown(f"""
        1. 영상 검색 {'✅' if current_step >= 1 else ''}
        2. 영상 선택 {'✅' if current_step >= 2 else ''}
        3. 아이디어 생성 {'✅' if current_step >= 3 else ''}
        """)
        
        if st.session_state.selected_videos:
            st.markdown("---")
            st.markdown("### 선택된 영상")
            for i, url in enumerate(st.session_state.selected_videos, 1):
                st.markdown(f"{i}. {url}")

    with st.container():
        st.header("🔍 영상 검색")
        with st.form(key='search_form'):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                search_keyword = st.text_input(
                    "검색어 입력",
                    placeholder="찾고 싶은 영상의 키워드를 입력하세요..."
                )
            with col2:
                duration_option = st.selectbox(
                    "영상 길이",
                    options=['any', 'very_short', 'short', 'medium', 'medium_long', 'long', 'very_long'],
                    format_func=lambda x: {
                        'any': '모든 길이',
                        'very_short': '1분 미만',
                        'short': '1-5분',
                        'medium': '5-15분',
                        'medium_long': '15-30분',
                        'long': '30-60분',
                        'very_long': '60분 이상'
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
            search_submitted = st.form_submit_button("검색하기", use_container_width=True)

        if search_submitted and search_keyword:
            with st.spinner('🔍 영상을 검색하고 있습니다...'):
                st.session_state.search_results = search_videos(search_keyword, duration_option, sort_option)
                st.session_state.search_performed = True

        if st.session_state.search_performed and st.session_state.search_results is not None:
            if not st.session_state.search_results.empty:
                for idx, video in st.session_state.search_results.iterrows():
                    with st.container():
                        cols = st.columns([4, 1])  # 4:1 비율로 분할
                        
                        with cols[0]:  # 왼쪽 컬럼 (비디오 정보)
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
                        
                        with cols[1]:  # 오른쪽 컬럼 (버튼)
                            if video['url'] not in st.session_state.selected_videos:
                                if st.button('선택', key=f"select_video_{video['video_id']}_{idx}"):
                                    st.session_state.selected_videos.append(video['url'])
                                    st.success(f"✅ 영상이 추가되었습니다!")
                            else:
                                st.warning("⚠️ 선택됨")
            else:
                st.info("🔍 검색 결과가 없습니다. 다른 검색어를 시도해보세요.")

        if st.session_state.selected_videos:
            st.markdown("---")
            st.header("📌 선택된 영상")
            
            for idx, url in enumerate(st.session_state.selected_videos):
                with st.container():
                    cols = st.columns([5, 1])
                    with cols[0]:
                        st.markdown(f"""
                        <div class="selected-video">
                            {idx + 1}. {url}
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        if st.button('제거', key=f'remove_video_{idx}'):
                            st.session_state.selected_videos.pop(idx)
                            st.rerun()

            if len(st.session_state.selected_videos) > 0:
                st.markdown("---")
                st.header("💡 아이디어 생성")
                
                # 사용자 프롬프트 입력 필드 추가
                user_prompt = st.text_area(
                    "프로젝트 요구사항 설명",
                    placeholder="어떤 프로젝트 아이디어가 필요하신가요? (예: B2B SaaS 솔루션, 소비자 앱, 교육 플랫폼 등)\n"
                              "목표하는 사용자층이나 특별히 고려해야 할 요구사항이 있다면 자세히 설명해주세요.",
                    help="AI가 더 정확한 아이디어를 생성하도록 프로젝트의 목적과 요구사항을 설명해주세요.",
                    height=150
                )
                
                if st.button('선택한 영상으로 아이디어 생성하기'):
                    if not user_prompt.strip():
                        st.warning("⚠️ 프로젝트 요구사항을 입력해주세요.")
                    else:
                        selected_videos = st.session_state.selected_videos
                        combined_transcript, generated_idea = generate_idea_from_videos(selected_videos, user_prompt)
                        
                        if generated_idea:
                            st.markdown("### 🎉 생성된 프로젝트 아이디어")
                            st.markdown(generated_idea)
                        else:
                            st.error("아이디어 생성에 실패했습니다.")

    if st.session_state.generated_idea:
        st.markdown("---")
        st.header("✨ 생성된 아이디어")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border: 1px solid #dee2e6;">
            {st.session_state.generated_idea}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button('새 프로젝트 시작하기', key='new_project_button_bottom', use_container_width=True):
            st.session_state.generated_idea = None
            st.session_state.selected_videos = []
            st.session_state.search_results = None
            st.session_state.search_performed = False
            st.rerun()

if __name__ == "__main__":
    main()