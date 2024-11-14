import streamlit as st
import whisper
import yt_dlp
import openai
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ideas
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         video_url TEXT,
         transcript TEXT,
         idea TEXT,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': 'temp_audio.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'temp_audio.mp3'

def transcribe_audio(audio_path):
    import torch
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    
    # GPU 사용 가능한 경우에만 fp16=True 설정
    if device == "cuda":
        result = model.transcribe(audio_path, fp16=True)
    else:
        result = model.transcribe(audio_path)
    
    return result["text"]

def generate_idea(transcript):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "영상 내용을 바탕으로 혁신적인 프로젝트 아이디어를 제안해주세요."},
            {"role": "user", "content": f"다음 영상 내용을 바탕으로 프로젝트 아이디어를 제안해주세요: {transcript}"}
        ]
    )
    return response.choices[0].message.content

def save_idea(video_url, transcript, idea):
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('INSERT INTO ideas (video_url, transcript, idea) VALUES (?, ?, ?)',
              (video_url, transcript, idea))
    conn.commit()
    conn.close()

def main():
    st.set_page_config(page_title="유튜브 프로젝트 아이디어 생성기")
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
            st.write(f"**영상 URL:** {idea[1]}")
            st.write(f"**아이디어:** {idea[3]}")
            st.write("---")
        conn.close()
    
    # 메인 화면
    video_url = st.text_input("YouTube URL을 입력하세요")
    
    if st.button("아이디어 생성"):
        with st.spinner("영상 처리 중..."):
            # 1. 오디오 다운로드
            audio_path = download_audio(video_url)
            
            # 2. 음성을 텍스트로 변환
            transcript = transcribe_audio(audio_path)
            st.text_area("영상 내용", transcript, height=400)
            
            # 3. 프로젝트 아이디어 생성
            idea = generate_idea(transcript)
            st.text_area("프로젝트 아이디어", idea, height=1000)
            
            # 4. 아이디어 저장
            save_idea(video_url, transcript, idea)
            
            # 임시 파일 삭제
            os.remove(audio_path)
            
            st.success("아이디어가 생성되고 저장되었습니다!")

if __name__ == "__main__":
    main()