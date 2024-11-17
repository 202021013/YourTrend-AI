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

def search_videos(keyword, duration_category):
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
            videos.append({
                'video_id': video['id'],
                'title': video['title'],
                'author': video['channel']['name'],
                'duration': video['duration'],
                'duration_seconds': video_duration_sec,
                'view_count': video['viewCount']['text'],
                'thumbnail': video['thumbnails'][0]['url'],
                'url': video['link']
            })
    return pd.DataFrame(videos)

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

def generate_idea(transcripts):
    try:
        combined_transcript = "\n".join(transcripts)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_transcript}
            ]
        )
        idea = response.choices[0].message.content
        return idea
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return ""

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

def main():
    st.set_page_config(
        page_title="유튜브 프로젝트 아이디어 생성기",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_videos' not in st.session_state:
        st.session_state.selected_videos = []
    if 'generated_idea' not in st.session_state:
        st.session_state.generated_idea = None
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

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
            col1, col2 = st.columns([3, 1])
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
            search_submitted = st.form_submit_button("검색하기", use_container_width=True)

        if search_submitted and search_keyword:
            with st.spinner('🔍 영상을 검색하고 있습니다...'):
                st.session_state.search_results = search_videos(search_keyword, duration_option)
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
                                st.success("✅ 영상이 선택되었습니다!")
                                st.balloons()
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
                        for i, video_url in enumerate(st.session_state.selected_videos):
                            audio_path = download_audio(video_url)
                            if audio_path:
                                transcript = transcribe_audio(audio_path)
                                if transcript:
                                    transcripts.append(transcript)
                                try:
                                    os.remove(audio_path)
                                except:
                                    pass
                            progress_bar.progress((i + 1) / len(st.session_state.selected_videos))
                        
                        if transcripts:
                            idea = generate_idea(transcripts)
                            st.session_state.generated_idea = idea
                            save_idea(st.session_state.selected_videos, "\n".join(transcripts), idea)
                            st.success("✨ 아이디어가 성공적으로 생성되었습니다!")
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