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
    
    @keyframes fade-in {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @keyframes slide-in {
        from {transform: translateX(-100%);}
        to {transform: translateX(0);}
    }

    .success-message {
        animation: fade-in 0.5s ease-out;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }

    .slide-effect {
        animation: slide-in 0.5s ease-out;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #cce5ff;
        color: #004085;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

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

def generate_idea(transcripts, video_urls):
    try:
        # 영상 콘텐츠 결합
        combined_transcript = "".join([
            f"\n[영상 {i}] URL: {url}\n===== 스크립트 시작 =====\n{transcript}\n===== 스크립트 끝 =====\n{'-' * 50}\n"
            for i, (transcript, url) in enumerate(zip(transcripts, video_urls), 1)
        ])
        
        system_prompt = """당신은 혁신적인 프로젝트 아이디어를 제안하는 전문가입니다. 
제공된 YouTube 영상 내용을 분석하여 실현 가능하고 창의적인 프로젝트 아이디어를 생성해야 합니다.

다음 형식으로 응답해주세요:

1. 프로젝트 개요
- 프로젝트 명칭
- 핵심 목표
- 주요 특징 (3-4개)

2. 세부 구현 방안
- 필요한 기술 스택
- 주요 기능 명세
- 개발 난이도 (상/중/하)

3. 차별화 포인트
- 기존 솔루션과의 차이점
- 독창적인 특징

4. 수익화 전략
- 목표 사용자층
- 수익 모델 제안
- 마케팅 전략

5. 개발 로드맵
- 1단계: MVP (Minimum Viable Product)
- 2단계: 핵심 기능 구현
- 3단계: 고도화 및 최적화"""

        user_prompt = f"다음은 YouTube 영상들의 내용을 텍스트로 변환한 것입니다.\n이 내용을 바탕으로 프로젝트 아이디어를 생성해주세요:\n\n{combined_transcript}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content, combined_transcript
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return "", ""

def save_idea(video_urls, transcript, idea):
    try:
        conn = sqlite3.connect('project_ideas.db')
        c = conn.cursor()
        urls_str = ", ".join(video_urls)
        c.execute('INSERT INTO ideas (video_urls, transcript, idea) VALUES (?, ?, ?)',
                 (urls_str, transcript, idea))
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

def generate_idea_from_videos(selected_videos):
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

        # 스크립트 결합 및 아이디어 생성
        combined_transcript = "\n".join(transcripts)
        
        system_prompt = """당신은 혁신적인 프로젝트 아이디어를 제안하는 전문가입니다. 
제공된 YouTube 영상 내용을 분석하여 실현 가능하고 창의적인 프로젝트 아이디어를 생성해야 합니다.

다음 형식으로 응답해주세요:

1. 프로젝트 개요
- 프로젝트 명칭
- 핵심 목표
- 주요 특징 (3-4개)

2. 세부 구현 방안
- 필요한 기술 스택
- 주요 기능 명세
- 개발 난이도 (상/중/하)

3. 차별화 포인트
- 기존 솔루션과의 차이점
- 독창적인 특징

4. 수익화 전략
- 목표 사용자층
- 수익 모델 제안
- 마케팅 전략

5. 개발 로드맵
- 1단계: MVP (Minimum Viable Product)
- 2단계: 핵심 기능 구현
- 3단계: 고도화 및 최적화"""

        user_prompt = f"""다음은 YouTube 영상들의 내용을 텍스트로 변환한 것입니다. 
이 내용을 바탕으로 프로젝트 아이디어를 생성해주세요:

{combined_transcript}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        generated_idea = response.choices[0].message.content
        return combined_transcript, generated_idea
        
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return None, None

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
                        st.markdown(f"""
                        <div class="video-card">
                            <div style="display: flex; align-items: center;">
                                <img src="{video['thumbnail']}" style="width: 200px; border-radius: 10px;"/>
                                <div style="margin-left: 20px;">
                                    <h3>{video['title']}</h3>
                                    <p>👤 {video['author']}</p>
                                    <p>⏱️ {video['duration']} | 👁️ {format_views(video['view_count'])}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if video['url'] not in st.session_state.selected_videos:
                            if st.button('영상 선택하기', key=f"select_video_{video['video_id']}_{idx}", use_container_width=True):
                                st.session_state.selected_videos.append(video['url'])
                                
                                # CSS 애니메이션 효과
                                st.markdown("""
                                    <div class="success-message">
                                        ✅ 영상이 성공적으로 선택되었습니다!
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("""
                                    <div class="slide-effect">
                                        🎉 새로운 영상이 추가되었습니다.
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Streamlit 내장 효과
                                #st.snow()  # 또는 st.balloons()
                                
                                # 추가 정보 표시
                                with st.spinner('영상 정보를 불러오는 중...'):
                                    time.sleep(0.5)  # 잠시 대기
                                st.success(f"'{video['title']}' 영상이 프로젝트에 추가되었습니다!")
                        else:
                            st.warning("⚠️ 이미 선택된 영상입니다!")
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
                if st.button('선택한 영상으로 아이디어 생성하기', key='generate_idea_button', use_container_width=True):
                    with st.spinner("🤖 영상을 분석하고 아이디어를 생성하고 있습니다..."):
                        transcripts = []
                        progress_bar = st.progress(0)
                        
                        # 영상 처리
                        for i, video_url in enumerate(st.session_state.selected_videos):
                            audio_path = download_audio(video_url)
                            if audio_path:
                                if transcript := transcribe_audio(audio_path):
                                    transcripts.append(transcript)
                                try:
                                    os.remove(audio_path)
                                except Exception:
                                    pass
                            progress_bar.progress((i + 1) / len(st.session_state.selected_videos))
                        
                        if transcripts:
                            idea, formatted_transcript = generate_idea(transcripts, st.session_state.selected_videos)
                            st.session_state.update({
                                'generated_idea': idea,
                                'transcript': formatted_transcript
                            })
                            save_idea(st.session_state.selected_videos, formatted_transcript, idea)
                            
                            st.success("✨ 아이디어가 성공적으로 생성되었습니다!")
                            with st.expander("📝 영상 내용 보기"):
                                st.text_area("영상 스크립트", st.session_state.transcript, height=400)
                        else:
                            st.error("❌ 선택된 영상에서 텍스트를 추출할 수 없습니다.")

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