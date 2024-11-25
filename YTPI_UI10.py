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
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from datetime import datetime

load_dotenv()

def clean_view_count(view_data: dict) -> int:
    """조회수 데이터에서 숫자 추출"""
    try:
        if isinstance(view_data, dict):
            view_text = view_data.get('short', '0')
        else:
            view_text = str(view_data)

        # 숫자와 K, M, B만 추출
        number = ''.join(filter(lambda x: x.isdigit() or x in 'KMB.', view_text.upper()))
        
        if not number:
            return 0

        # K, M, B에 따른 승수 계산
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
    """
    주어진 텍스트를 완전한 문장으로 끝나도록 잘라냅니다.
    
    Args:
        text (str): 원본 텍스트
        max_tokens (int): 최대 토큰 수
        
    Returns:
        str: 완전한 문장으로 끝나는 잘린 텍스트
    """
    # 텍스트를 토큰으로 변환 (간단한 근사치 계산: 영어 기준 1단어 = 1.3토큰)
    estimated_tokens = len(text.split()) * 1.3
    
    # 토큰 수가 제한을 넘지 않으면 전체 텍스트 반환
    if estimated_tokens <= max_tokens:
        return text
        
    # 대략적인 문자 수 계산 (토큰당 평균 4글자로 가정)
    approx_chars = int(max_tokens * 4)
    
    # 문장 끝 구분자 정의
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    
    # 대략적인 위치에서 시작하여 가장 가까운 문장 끝 찾기
    truncated_text = text[:approx_chars]
    
    # 가장 마지막 완전한 문장 찾기
    last_sentence_end = -1
    for ending in sentence_endings:
        pos = truncated_text.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos
            
    # 완전한 문장이 발견되면 해당 위치까지 자르기
    if last_sentence_end != -1:
        return text[:last_sentence_end + 2].strip()  # +2는 구분자 포함
    
    # 문장 끝을 찾지 못한 경우, 마지막 공백에서 자르기
    last_space = truncated_text.rfind(' ')
    if last_space != -1:
        return text[:last_space].strip() + "..."
        
    # 아무 것도 찾지 못한 경우 그냥 자르고 ... 추가
    return truncated_text.strip() + "..."

class YourClassName:
    def __init__(self, name, role, personality, client, temperature=0.7):
        self.name = name
        self.role = role
        self.personality = personality
        self.client = client
        self.temperature = temperature
        self.conversation_history = []

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

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": truncate_to_complete_sentence(topic[:2000], 500)}  # 토큰 제한 적용
        ]
        
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
            
        messages.extend(self.conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=200 if round_num > 1 else 1000,
                temperature=self.temperature
            )
            
            generated_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            
            return generated_response
        
        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"

def search_videos(keyword: str, duration: str = 'any', sort: str = 'relevance') -> pd.DataFrame:
    """유튜��� 영상 검색 및 결과 반환"""
    try:
        videos_search = VideosSearch(keyword, limit=10)
        search_result = videos_search.result()
        
        if not search_result or 'result' not in search_result:
            return pd.DataFrame()
            
        results = []
        
        for video in search_result['result']:
            try:
                # 영상 길이 파싱
                duration_str = video.get('duration', '0:00')
                duration_parts = duration_str.split(':')
                total_minutes = 0
                
                if len(duration_parts) == 2:  # MM:SS
                    total_minutes = int(duration_parts[0])
                elif len(duration_parts) == 3:  # HH:MM:SS
                    total_minutes = int(duration_parts[0]) * 60 + int(duration_parts[1])
                
                # 길이 필터링
                if duration == 'short' and total_minutes > 5:
                    continue
                elif duration == 'medium' and (total_minutes <= 5 or total_minutes > 15):
                    continue
                elif duration == 'long' and total_minutes <= 15:
                    continue
                
                # 조회수 처리
                view_count = clean_view_count(video.get('viewCount', {}))
                
                # 썸네일 처리
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
        
        # 정렬
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
    """조회수를 읽기 쉬운 형식으로 변환"""
    try:
        if not isinstance(view_count, (int, float)):
            return "0"
            
        if view_count >= 1000000000:  # Billions
            return f"{view_count/1000000000:.1f}B"
        elif view_count >= 1000000:    # Millions
            return f"{view_count/1000000:.1f}M"
        elif view_count >= 1000:       # Thousands
            return f"{view_count/1000:.1f}K"
        return str(view_count)
    except:
        return "0"

# 검색 결과 표시 부분도 수정
def display_video_result(video: pd.Series):
    """비디오 검색 결과를 표시"""
    try:
        formatted_views = format_views(video['view_count'])
        
        return f"""
        <div class="video-card">
            <div style="display: flex; align-items: start;">
                <img src="{video['thumbnail']}" style="width: 200px; border-radius: 10px;"/>
                <div style="margin-left: 20px; flex-grow: 1;">
                    <h3>{video['title']}</h3>
                    <p>👤 {video['author']}</p>
                    <p>⏱️ {video['duration']} | 👁️ {formatted_views}</p>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        st.error(f"비디오 표시 중 오류 발생: {str(e)}")
        return ""

def download_audio(video_url: str) -> str:
    """유튜브 영상에서 오디오 추출"""
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

def transcribe_audio(audio_path: str) -> str:
    """오디오 파일을 텍스트로 변환"""
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        return result["text"]
        
    except Exception as e:
        st.error(f"음성 인식 중 오류 발생: {str(e)}")
        return None

class AIAgent:
    def __init__(self, name: str, role: str, temperature: float, personality: str):
        self.name = name
        self.role = role
        self.temperature = temperature
        self.personality = personality
        self.client = OpenAI()
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
    
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": truncate_to_complete_sentence(topic[:2000], 500)}  # 토큰 제한 적용
        ]
        
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
            
        messages.extend(self.conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=200 if round_num > 1 else 1000,
                temperature=self.temperature
            )
            
            generated_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            
            return generated_response
        
        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"
            
    def format_conversation_history(self) -> str:
        formatted = []
        for msg in self.conversation_history:
            formatted.append(f"{self.name}: {msg['content']}")
        return "\n\n".join(formatted)

def generate_discussion(transcripts: List[str], video_urls: List[str], user_prompt: str) -> tuple:
    # AI 에이전트 초기화
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

def generate_final_summary(conversation: List[dict], user_prompt: str) -> str:
    client = OpenAI()
    
    conversation_summary = "\n\n".join([
        f"라운드 {msg['round']} - {msg['agent']}: {msg['response'][:500]}"
        for msg in conversation
    ])
    
    prompt = f"""
주제: {user_prompt}

전문가들의 논의 내용을 바탕으로 다음 형식으로 최종 프로젝트 제안서를 작성해주세요:

1. 프로젝트 개요
   - 핵심 가치 제안
   - 목표 시장 및 사용자

2. 핵심 기능 및 특징
   - 주요 기능
   - 차별화 요소

3. 기술 구현 방안
   - 사용 기술 스택
   - 개발 요구사항

4. 비즈니스 전략
   - 수익 모델
   - 마케팅 전략

5. 프로젝트 로드맵
   - 단계별 목표
   - 주요 마일스톤

6. 위험 요소 및 대응 방안

전문가 논의 내용:
{conversation_summary}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"최종 요약 생성 중 오류 발생: {str(e)}"

def create_context(transcripts: List[str], video_urls: List[str]) -> str:
    return "".join([
        f"\n[영상 {i+1}] URL: {url}\n영상 내용 요약:\n{transcript}\n{'-'*50}"
        for i, (transcript, url) in enumerate(zip(transcripts, video_urls))
    ])

def init_db():
    conn = sqlite3.connect('project_ideas.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS ideas')
    c.execute('''
        CREATE TABLE ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            conversation_history TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_idea(video_urls: List[str], conversation_history: List[dict], final_summary: str):
    try:
        conn = sqlite3.connect('project_ideas.db')
        c = conn.cursor()
        urls_str = ", ".join(video_urls)
        conversation_str = json.dumps(conversation_history, ensure_ascii=False)
        c.execute('INSERT INTO ideas (video_urls, conversation_history, final_summary) VALUES (?, ?, ?)', 
                 (urls_str, conversation_str, final_summary))
        conn.commit()
        conn.close()
    except Exception:
        pass
def save_idea(video_urls: List[str], conversation_history: List[dict], final_summary: str):
    try:
        conn = sqlite3.connect('project_ideas.db')
        c = conn.cursor()
        urls_str = ", ".join(video_urls)
        conversation_str = json.dumps(conversation_history, ensure_ascii=False)
        c.execute('INSERT INTO ideas (video_urls, conversation_history, final_summary) VALUES (?, ?, ?)', 
                 (urls_str, conversation_str, final_summary))
        conn.commit()
        conn.close()
        st.success("✅ 아이디어가 성공적으로 저장되었습니다!")
    except Exception as e:
        st.error(f"아이디어 저장 중 오류 발생: {str(e)}")

def generate_idea_from_videos(selected_videos: List[str], user_prompt: str):
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

        # AI 에이전트 토론 시작
        st.markdown("### 🤖 AI 전문가 토론 시작")
        final_summary, conversation = generate_discussion(transcripts, selected_videos, user_prompt)
            
        return final_summary, conversation
        
    except Exception as e:
        st.error(f"아이디어 생성 중 오류 발생: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="유튜브 프로젝트 아이디어 생성기",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 세션 상태 초기화
    if 'selected_videos' not in st.session_state:
        st.session_state.selected_videos = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

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

    # 데이터베이스 초기화
    init_db()

    # 메인 컨테이너
    with st.container():
        # 영상 검색 섹션
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
                            
                            # 결과 저장
                            save_idea(
                                st.session_state.selected_videos,
                                conversation_history,
                                final_summary
                            )

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