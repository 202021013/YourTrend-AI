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
from datetime import datetime

# .env 파일 로드
load_dotenv()

# 카테고리 정의
CATEGORIES = {
    'technology': ['프로그래밍', '인공지능', '웹개발', '모바일앱', '데이터사이언스', 'IoT', '클라우드'],
    'business': ['스타트업', '마케팅', '창업', '비즈니스모델', '투자', '금융'],
    'design': ['UI/UX', '그래픽디자인', '제품디자인', '브랜딩', '모션그래픽'],
    'education': ['이러닝', '교육테크', '학습도구', '교육컨텐츠', '온라인강의'],
    'entertainment': ['게임', '미디어', '스트리밍', '콘텐츠제작', '엔터테인먼트'],
    'health': ['헬스케어', '피트니스', '웰니스', '의료기술', '건강관리'],
    'social': ['소셜미디어', '커뮤니티', '소셜임팩트', '공유경제', '협업도구']
}

# SQLite DB 초기화 (테이블 구조 개선)
def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_urls TEXT,
            transcript TEXT,
            idea TEXT,
            category TEXT,
            subcategory TEXT,
            keywords TEXT,
            difficulty_level TEXT,
            estimated_duration TEXT,
            required_skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def categorize_content(transcript):
    """트랜스크립트 내용을 분석하여 적절한 카테고리와 키워드 추출"""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                다음 카테고리 중에서 가장 적절한 메인 카테고리와 서브카테고리를 선택하고,
                관련 키워드를 추출해주세요:
                
                카테고리: technology, business, design, education, entertainment, health, social
                
                결과는 다음 형식으로 반환해주세요:
                {
                    "main_category": "선택된 메인 카테고리",
                    "subcategory": "선택된 서브카테고리",
                    "keywords": ["키워드1", "키워드2", "키워드3"],
                    "difficulty": "초급/중급/고급",
                    "duration": "예상 개발 기간",
                    "required_skills": ["필요기술1", "필요기술2"]
                }
                """},
                {"role": "user", "content": f"다음 내용을 분석해주세요: {transcript}"}
            ]
        )
        return eval(response.choices[0].message.content)
    except Exception as e:
        st.error(f"컨텐츠 분류 중 오류 발생: {str(e)}")
        return {
            "main_category": "uncategorized",
            "subcategory": "general",
            "keywords": [],
            "difficulty": "미정",
            "duration": "미정",
            "required_skills": []
        }

def save_idea(video_urls, transcript, idea, category_info):
    """아이디어와 분류 정보를 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect('project_ideas.db')
        c = conn.cursor()
        urls_str = ", ".join(video_urls)
        c.execute('''
            INSERT INTO ideas (
                video_urls, transcript, idea, category, subcategory,
                keywords, difficulty_level, estimated_duration, required_skills
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            urls_str, transcript, idea,
            category_info['main_category'],
            category_info['subcategory'],
            ",".join(category_info['keywords']),
            category_info['difficulty'],
            category_info['duration'],
            ",".join(category_info['required_skills'])
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"아이디어 저장 중 오류 발생: {str(e)}")

def search_ideas(keyword=None, category=None, difficulty=None):
    """저장된 아이디어 검색 (필터링 기능 추가)"""
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    
    query = "SELECT * FROM ideas WHERE 1=1"
    params = []
    
    if keyword:
        query += " AND (idea LIKE ? OR keywords LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if difficulty:
        query += " AND difficulty_level = ?"
        params.append(difficulty)
    
    query += " ORDER BY created_at DESC"
    
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    return results

def generate_enhanced_idea(transcripts, category_info):
    """카테고리 정보를 활용하여 더 구체적인 아이디어 생성"""
    try:
        client = openai.OpenAI()
        combined_transcript = "\n".join(transcripts)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""
                다음 조건을 고려하여 혁신적인 프로젝트 아이디어를 제안해주세요:
                - 카테고리: {category_info['main_category']}
                - 서브카테고리: {category_info['subcategory']}
                - 난이도: {category_info['difficulty']}
                - 필요 기술: {', '.join(category_info['required_skills'])}
                
                다음 형식으로 답변해주세요:
                1. 프로젝트 개요
                2. 주요 기능
                3. 기술 스택
                4. 개발 단계
                5. 예상 소요 기간
                6. 확장 가능성
                """},
                {"role": "user", "content": f"다음 내용을 바탕으로 프로젝트 아이디어를 제안해주세요: {combined_transcript}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return ""

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
                'video_id': video_id,
                'upload_date': result.get('upload_date', "날짜 정보 없음"),
                'description': result.get('description', "설명 없음")
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
            'video_id': video_id,
            'upload_date': "날짜 정보 없음",
            'description': "설명 없음"
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
    """YouTube 영상에서 오디오 추출"""
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
    """Whisper를 사용하여 오디오를 텍스트로 변환"""
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

async def process_video_async(video, temp_dir):
    """비디오 처리를 비동기적으로 수행"""
    video_id = video['video_id']
    audio_path = os.path.join(temp_dir, f"temp_audio_{video_id}.mp3")
    
    if video_id in st.session_state.processed_videos:
        return st.session_state.current_transcripts.get(video_id)
    
    try:
        audio_file = await asyncio.to_thread(download_audio, video['url'])
        if (audio_file and os.path.exists(audio_file)):
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

def export_results(results):
    """검색 결과를 CSV 또는 JSON으로 내보내기"""
    df = pd.DataFrame(results, columns=[
        'id', 'video_urls', 'transcript', 'idea', 'category',
        'subcategory', 'keywords', 'difficulty_level',
        'estimated_duration', 'required_skills', 'created_at'
    ])
    
    export_format = st.radio("내보내기 형식 선택:", ['CSV', 'JSON'])
    if export_format == 'CSV':
        csv = df.to_csv(index=False)
        st.download_button(
            "CSV 다운로드",
            csv,
            "project_ideas.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        json = df.to_json(orient='records')
        st.download_button(
            "JSON 다운로드",
            json,
            "project_ideas.json",
            "application/json",
            key='download-json'
        )

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

    # 사이드바에 검색 필터 추가
    st.sidebar.title("검색 필터")
    search_type = st.sidebar.radio("검색 유형", ["새 아이디어 생성", "저장된 아이디어 검색"])
    
    if search_type == "새 아이디어 생성":
        # 메인 카테고리 선택
        selected_category = st.sidebar.selectbox(
            "카테고리 선택",
            list(CATEGORIES.keys()),
            format_func=lambda x: x.capitalize()
        )
        
        # 선택된 카테고리의 서브카테고리 표시
        selected_subcategories = st.sidebar.multiselect(
            "서브카테고리 선택",
            CATEGORIES[selected_category]
        )
        
        # 난이도 선택
        difficulty_level = st.sidebar.select_slider(
            "프로젝트 난이도",
            options=["초급", "중급", "고급"]
        )

        # 유튜브 영상 검색
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
                            audio_path = os.path.join("temp_audio_files", f"temp_audio_{video['video_id']}.mp3")
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

                        if all_transcripts:
                            combined_transcript = "\n".join(all_transcripts)
                            category_info = categorize_content(combined_transcript)
                            
                            st.subheader("컨텐츠 분석 결과")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"메인 카테고리: {category_info['main_category']}")
                                st.write(f"서브카테고리: {category_info['subcategory']}")
                                st.write(f"난이도: {category_info['difficulty']}")
                            with col2:
                                st.write(f"키워드: {', '.join(category_info['keywords'])}")
                                st.write(f"예상 개발 기간: {category_info['duration']}")
                                st.write(f"필요 기술: {', '.join(category_info['required_skills'])}")

                            idea = generate_enhanced_idea(all_transcripts, category_info)
                            if idea:
                                st.subheader("생성된 아이디어")
                                st.write(idea)
                                
                                save_idea(
                                    [v['url'] for v in selected_videos],
                                    combined_transcript,
                                    idea,
                                    category_info
                                )
                                st.success("아이디어가 생성되고 저장되었습니다!")
                        else:
                            st.error("선택한 영상에서 텍스트를 추출할 수 없습니다.")

    else:  # 저장된 아이디어 검색
        search_keyword = st.sidebar.text_input("키워드 검색")
        category_filter = st.sidebar.selectbox(
            "카테고리 필터",
            ["전체"] + list(CATEGORIES.keys()),
            format_func=lambda x: x.capitalize()
        )
        difficulty_filter = st.sidebar.selectbox(
            "난이도 필터",
            ["전체", "초급", "중급", "고급"]
        )
        
        # 검색 실행
        search_results = search_ideas(
            keyword=search_keyword if search_keyword else None,
            category=category_filter if category_filter != "전체" else None,
            difficulty=difficulty_filter if difficulty_filter != "전체" else None
        )
        
        if search_results:
            st.subheader("검색 결과")
            for result in search_results:
                with st.expander(f"아이디어: {result[3][:100]}..."):
                    st.write(f"카테고리: {result[4]}")
                    st.write(f"서브카테고리: {result[5]}")
                    st.write(f"키워드: {result[6]}")
                    st.write(f"난이도: {result[7]}")
                    st.write(f"예상 개발 기간: {result[8]}")
                    st.write(f"필요 기술: {result[9]}")
                    st.write("전체 아이디어:")
                    st.write(result[3])
        else:
            st.info("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()