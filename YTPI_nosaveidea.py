import streamlit as st
import whisper
import yt_dlp
import openai
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
from pytube import Search, YouTube
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# .env 파일 로드
load_dotenv()

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
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
    """YouTube URL에서 비디오 ID 추출"""
    import re
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_video_info(video):
    """yt-dlp를 사용하여 비디오 정보를 가져오는 함수"""
    try:
        video_id = video.video_id
        url = f"https://youtube.com/watch?v={video_id}"
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            return {
                'title': result.get('title', "제목 없음"),
                'url': url,
                'thumbnail': result.get('thumbnail', f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"),
                'duration': result.get('duration', 0),
                'view_count': result.get('view_count', 0),
                'author': result.get('uploader', "작성자 정보 없음"),
                'video_id': video_id
            }
    except Exception as e:
        st.warning(f"영상 정보를 가져오는 중 오류 발생: {str(e)}")
        return {
            'title': video.title or "제목 없음",
            'url': url,
            'thumbnail': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            'duration': 0,
            'view_count': 0,
            'author': video.author or "작성자 정보 없음",
            'video_id': video_id
        }

def search_videos(keyword, max_results=5):
    """키워드로 유튜브 영상 검색"""
    try:
        s = Search(keyword)
        videos = []
        for video in s.results[:max_results]:
            video_info = get_video_info(video)
            videos.append(video_info)
        return pd.DataFrame(videos)
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        return pd.DataFrame()

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
            'outtmpl': f'{output_path}.%(ext)s'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f'{output_path}.mp3'
    except Exception as e:
        st.error(f"오디오 다운로드 중 오류 발생: {str(e)}")
        return None

def transcribe_audio(audio_path):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        
        if device == "cuda":
            result = model.transcribe(audio_path, fp16=True)
        else:
            result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"음성 변환 중 오류 발생: {str(e)}")
        return ""

def generate_idea(transcripts):
    try:
        client = openai.OpenAI()
        combined_transcript = "\n".join(transcripts)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "선택된 여러 영상들의 내용을 종합하여 혁신적인 프로젝트 아이디어를 제안해주세요."},
                {"role": "user", "content": f"다음 영상들의 내용을 바탕으로 프로젝트 아이디어를 제안해주세요: {combined_transcript}"}
            ]
        )
        return response.choices[0].message.content
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

def format_duration(seconds):
    """초를 시:분:초 형식으로 변환"""
    if seconds == 0:
        return "길이 정보 없음"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"

def format_views(views):
    """조회수를 읽기 쉬운 형식으로 변환"""
    if views == 0:
        return "조회수 정보 없음"
    if views >= 1000000:
        return f"{views/1000000:.1f}M"
    if views >= 1000:
        return f"{views/1000:.1f}K"
    return str(views)

async def process_video_async(video, temp_dir):
    video_id = video['video_id']
    audio_path = os.path.join(temp_dir, f"temp_audio_{video_id}.mp3")
    
    if video_id in st.session_state.processed_videos:
        return st.session_state.current_transcripts.get(video_id)
    
    try:
        audio_file = await asyncio.to_thread(download_audio, video['url'])
        if audio_file and os.path.exists(audio_file):
            os.rename(audio_file, audio_path)
        
        model = whisper.load_model("tiny")
        transcript = await asyncio.to_thread(model.transcribe, audio_path)
        
        if transcript and transcript.get("text"):
            st.session_state.current_transcripts[video_id] = transcript["text"]
            st.session_state.processed_videos[video_id] = True
            
        try:
            os.remove(audio_path)
        except Exception:
            pass
            
        return transcript.get("text")
    except Exception as e:
        st.error(f"처리 중 오류 발생: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="유튜브 프로젝트 아이디어 생성기", layout="wide")
    st.title("유튜브 프로젝트 아이디어 생성기")

    # 세션 상태 초기화
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = {}
    if 'current_transcripts' not in st.session_state:
        st.session_state.current_transcripts = {}
    
    init_db()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    temp_dir = "temp_audio_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    search_keyword = st.text_input("검색어를 입력하세요")
    if search_keyword:
        videos_df = search_videos(search_keyword)
        if not videos_df.empty:
            st.subheader("검색 결과")
            selected_videos = []
            
            for idx, video in videos_df.iterrows():
                col_thumb, col_info = st.columns([1, 3])
                with col_thumb:
                    st.image(video['thumbnail'], use_column_width=True)
                with col_info:
                    st.write(f"**{video['title']}**")
                    st.write(f"작성자: {video['author']}")
                    st.write(f"길이: {format_duration(video['duration'])}")
                    st.write(f"조회수: {format_views(video['view_count'])}")
                    selected = st.checkbox("선택", key=f"video_{idx}")
                    if selected:
                        selected_videos.append(video)
                st.write("---")

            if st.button("선택한 영상으로 아이디어 생성", key="generate_idea_btn") and selected_videos:
                with st.spinner("영상 처리 중..."):
                    all_transcripts = []
                    
                    for video in selected_videos:
                        audio_path = os.path.join(temp_dir, f"temp_audio_{video['video_id']}.mp3")
                        audio_file = download_audio(video['url'])
                        
                        if audio_file:
                            if os.path.exists(audio_file):
                                os.rename(audio_file, audio_path)
                            
                            transcript = transcribe_audio(audio_path)
                            if transcript:
                                all_transcripts.append(transcript)
                            
                            try:
                                os.remove(audio_path)
                            except Exception as e:
                                st.warning(f"임시 파일 삭제 중 오류 발생: {str(e)}")

                    for file in os.listdir(temp_dir):
                        if file.startswith("temp_audio_"):
                            try:
                                os.remove(os.path.join(temp_dir, file))
                            except Exception as e:
                                st.warning(f"임시 파일 삭제 중 오류 발생: {str(e)}")

                    if all_transcripts:
                        st.subheader("영상 내용")
                        for i, transcript in enumerate(all_transcripts, 1):
                            with st.expander(f"영상 {i} 내용"):
                                st.write(transcript)

                        idea = generate_idea(all_transcripts)
                        if idea:
                            st.subheader("생성된 아이디어")
                            st.write(idea)
                            save_idea([v['url'] for v in selected_videos],
                                    "\n".join(all_transcripts), idea)
                            st.success("아이디어가 생성되고 저장되었습니다!")
                    else:
                        st.error("선택한 영상에서 텍스트를 추출할 수 없습니다.")

if __name__ == "__main__":
    main()