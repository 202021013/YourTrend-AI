import streamlit as st
import whisper
import yt_dlp
import openai
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
from pytube import Search
import pandas as pd

# .env 파일 로드
load_dotenv()

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ideas
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         video_urls TEXT,
         transcript TEXT,
         idea TEXT,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

def search_videos(keyword, max_results=5):
    """키워드로 유튜브 영상 검색"""
    s = Search(keyword)
    s.results
    videos = []
    for video in s.results[:max_results]:
        videos.append({
            'title': video.title,
            'url': f"https://youtube.com/watch?v={video.video_id}",
            'thumbnail': video.thumbnail_url,
            'duration': video.length,
            'view_count': video.views,
            'author': video.author
        })
    return pd.DataFrame(videos)

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': f'temp_audio_{Path(url).stem}.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f'temp_audio_{Path(url).stem}.mp3'

def transcribe_audio(audio_path):
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    
    if device == "cuda":
        result = model.transcribe(audio_path, fp16=True)
    else:
        result = model.transcribe(audio_path)
    
    return result["text"]

def generate_idea(transcripts):
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

def save_idea(video_urls, transcript, idea):
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    urls_str = ", ".join(video_urls)
    c.execute('INSERT INTO ideas (video_urls, transcript, idea) VALUES (?, ?, ?)',
              (urls_str, transcript, idea))
    conn.commit()
    conn.close()

def main():
    st.set_page_config(page_title="유튜브 프로젝트 아이디어 생성기", layout="wide")
    st.title("유튜브 프로젝트 아이디어 생성기")
    
    # DB 초기화
    init_db()
    
    # OpenAI API 키 설정
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # 사이드바 - 저장된 아이디어
    with st.sidebar:
        st.header("저장된 아이디어")
        conn = sqlite3.connect('project_ideas.db')
        ideas = conn.execute('SELECT * FROM ideas ORDER BY created_at DESC').fetchall()
        for idea in ideas:
            st.write(f"**영상 URLs:** {idea[1]}")
            with st.expander("아이디어 보기"):
                st.write(idea[3])
            st.write("---")
        conn.close()
    
    # 메인 화면
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_keyword = st.text_input("검색어를 입력하세요")
        if search_keyword:
            videos_df = search_videos(search_keyword)
            
            # 검색 결과 표시
            st.subheader("검색 결과")
            selected_videos = []
            for idx, video in videos_df.iterrows():
                col_thumb, col_info = st.columns([1, 3])
                with col_thumb:
                    st.image(video['thumbnail'], use_column_width=True)
                with col_info:
                    st.write(f"**{video['title']}**")
                    st.write(f"작성자: {video['author']}")
                    st.write(f"조회수: {video['view_count']:,}")
                    selected = st.checkbox("선택", key=f"video_{idx}")
                    if selected:
                        selected_videos.append(video['url'])
                st.write("---")
    
    with col2:
        if st.button("선택한 영상으로 아이디어 생성") and selected_videos:
            with st.spinner("영상 처리 중..."):
                all_transcripts = []
                
                # 선택된 모든 영상 처리
                for url in selected_videos:
                    # 1. 오디오 다운로드
                    audio_path = download_audio(url)
                    
                    # 2. 음성을 텍스트로 변환
                    transcript = transcribe_audio(audio_path)
                    all_transcripts.append(transcript)
                    
                    # 임시 파일 삭제
                    os.remove(audio_path)
                
                # 모든 트랜스크립트 표시
                st.subheader("영상 내용")
                for i, transcript in enumerate(all_transcripts, 1):
                    with st.expander(f"영상 {i} 내용"):
                        st.write(transcript)
                
                # 3. 프로젝트 아이디어 생성
                combined_transcript = "\n".join(all_transcripts)
                idea = generate_idea(all_transcripts)
                st.subheader("생성된 아이디어")
                st.write(idea)
                
                # 4. 아이디어 저장
                save_idea(selected_videos, combined_transcript, idea)
                
                st.success("아이디어가 생성되고 저장되었습니다!")

if __name__ == "__main__":
    main()