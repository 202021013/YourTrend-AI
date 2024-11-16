import streamlit as st
import whisper
import yt_dlp
from openai import OpenAI
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
from pytube import Search, YouTube
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from youtubesearchpython import VideosSearch

# .env 파일 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    # 기존 테이블이 있다면 삭제
    c.execute('DROP TABLE IF EXISTS ideas')
    # 새로운 스키마로 테이블 생성
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
    """유튜브 URL에서 비디오 ID 추출"""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def safe_int_conversion(value, default=0):
    """안전하게 정수로 변환하는 헬퍼 함수"""
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
    """영상 길이 문자열을 초 단위로 변환"""
    if not duration_text:
        return 0
        
    try:
        parts = duration_text.split(':')
        parts = [int(p) for p in parts]
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 1:  # SS
            return parts[0]
        else:
            return 0
    except:
        return 0

def search_videos(keyword, duration_category):
    """유튜브 비디오 검색"""
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
                {"role": "system", "content": "선택된 여러 영상들의 내용을 종합하여 혁신적인 프로젝트 아이디어를 제안해주세요."},
                {"role": "user", "content": f"다음 영상들의 내용을 바탕으로 프로젝트 아이디어를 제안해주세요: {combined_transcript}"}
            ]
        )
        return response.choices[0].message.content.strip()
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
    """조회수를 읽기 쉬운 형식으로 변환"""
    if not view_count:
        return "조회수 정보 없음"
    try:
        view_count = view_count.replace('views', '').strip()
        if 'K' in view_count:
            return view_count
        elif 'M' in view_count:
            return view_count
        else:
            count = int(view_count.replace(',', ''))
            if count >= 1000000:
                return f"{count/1000000:.1f}M"
            elif count >= 1000:
                return f"{count/1000:.1f}K"
            else:
                return str(count)
    except:
        return "조회수 정보 없음"

def main():
    st.set_page_config(page_title="유튜브 프로젝트 아이디어 생성기", layout="wide")
    st.title("유튜브 프로젝트 아이디어 생성기")
    init_db()

    # 세션 상태 초기화
    if 'selected_videos' not in st.session_state:
        st.session_state.selected_videos = []
    if 'generated_idea' not in st.session_state:
        st.session_state.generated_idea = None

    # 검색 섹션
    with st.container():
        st.header("1️⃣ 영상 검색")
        col1, col2 = st.columns(2)
        with col1:
            search_keyword = st.text_input("검색어를 입력하세요")
        
        col3, col4 = st.columns(2)
        with col3:
            duration_option = st.selectbox(
                "영상 길이",
                options=['any', 'very_short', 'short', 'medium', 'medium_long', 'long', 'very_long'],
                format_func=lambda x: {
                    'any': '전체',
                    'very_short': '1분 이하',
                    'short': '1-5분',
                    'medium': '5-15분',
                    'medium_long': '15-30분',
                    'long': '30분-1시간',
                    'very_long': '1시간 이상'
                }[x]
            )

    # 검색 결과 표시
    if search_keyword:
        with st.spinner('검색 중... 잠시만 기다려주세요.'):
            try:
                videos = search_videos(keyword=search_keyword, duration_category=duration_option)
                
                if not videos.empty:
                    for idx, video in videos.iterrows():
                        with st.container():
                            st.write("---")
                            cols = st.columns([2, 3, 1])
                            
                            with cols[0]:
                                if video['thumbnail']:
                                    st.image(video['thumbnail'], use_column_width=True)
                            
                            with cols[1]:
                                st.markdown(f"**{video['title']}**")
                                st.write(f"👤 {video['author']}")
                                st.write(f"⏱️ {video['duration']}")
                                st.write(f"👁️ {format_views(video['view_count'])}")
                            
                            with cols[2]:
                                if st.button('선택', key=f"select_{video['video_id']}_{idx}", use_container_width=True):
                                    if video['url'] not in st.session_state.selected_videos:
                                        st.session_state.selected_videos.append(video['url'])
                                        st.success("영상이 선택되었습니다!")
                                    else:
                                        st.warning("이미 선택된 영상입니다!")
                else:
                    st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 선택된 영상 목록과 아이디어 생성 섹션
    if st.session_state.selected_videos:
        st.header("2️⃣ 선택된 영상")
        selected_df = pd.DataFrame([{'url': url} for url in st.session_state.selected_videos])
        
        for idx, row in selected_df.iterrows():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"{idx + 1}. {row['url']}")
            with col2:
                if st.button('제거', key=f'remove_{idx}'):
                    st.session_state.selected_videos.pop(idx)
                    st.experimental_rerun()
        
        st.header("3️⃣ 아이디어 생성")
        if st.button('선택한 영상으로 아이디어 생성하기', use_container_width=True):
            with st.spinner("선택된 영상들을 분석하여 아이디어를 생성하고 있습니다..."):
                try:
                    transcripts = []
                    for video_url in st.session_state.selected_videos:
                        audio_path = download_audio(video_url)
                        if audio_path:
                            transcript = transcribe_audio(audio_path)
                            if transcript:
                                transcripts.append(transcript)
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                    
                    if transcripts:
                        idea = generate_idea(transcripts)
                        st.session_state.generated_idea = idea
                        save_idea(st.session_state.selected_videos, "\n".join(transcripts), idea)
                    else:
                        st.error("선택된 영상에서 텍스트를 추출할 수 없습니다.")
                except Exception as e:
                    st.error(f"아이디어 생성 중 오류가 발생했습니다: {str(e)}")
        
        # 생성된 아이디어 표시
        if st.session_state.generated_idea:
            st.subheader("생성된 아이디어")
            st.write(st.session_state.generated_idea)
            
            if st.button('새로운 아이디어 생성하기'):
                st.session_state.generated_idea = None
                st.session_state.selected_videos = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()