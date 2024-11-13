import streamlit as st
from langchain_openai import ChatOpenAI  # 수정된 임포트
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import yt_dlp
import requests
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import isodate
from dataclasses import dataclass
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

@dataclass
class VideoInfo:
    title: str
    url: str
    duration: int
    thumbnail_url: str
    description: str
    views: int
    published_at: datetime
    
    @property
    def duration_str(self) -> str:
        hours = self.duration // 3600
        minutes = (self.duration % 3600) // 60
        seconds = self.duration % 60
        parts = []
        if hours > 0:
            parts.append(f"{hours}시간")
        if minutes > 0:
            parts.append(f"{minutes}분")
        if seconds > 0:
            parts.append(f"{seconds}초")
        return " ".join(parts)

class YouTubeIdeaGenerator:
    def __init__(self, openai_api_key: str, youtube_api_key: str):
        # YouTube API 키를 인스턴스 변수로 저장
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        
        # ChatGPT 초기화
        self.chat_model = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # 프롬프트 템플릿 업데이트
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 창의적인 프로젝트 아이디어를 제안하는 AI 멘토입니다.
            주어진 YouTube 영상 내용을 바탕으로 실현 가능한 프로젝트 아이디어를 제시해주세요."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # 메모리 설정
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # Runnable 체인 설정
        self.chain = RunnableWithMessageHistory(
            self.prompt | self.chat_model,
            lambda session_id: self.memory,
            input_messages_key="input",
            history_messages_key="history"
        )

    def search_videos(self, query: str, max_results: int = 5, 
                     min_duration: int = 0, max_duration: int = 1200) -> List[VideoInfo]:
        """지정된 길이 범위 내의 YouTube 영상 검색"""
        try:
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                params = {
                    "part": "snippet",
                    "q": query,
                    "maxResults": min(max_results * 2, 50),
                    "type": "video",
                    "key": self.youtube_api_key,
                    "pageToken": next_page_token
                }
                
                response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
                if response.status_code == 403:
                    logger.error("YouTube API 요청이 금지되었습니다. API 키 권한을 확인해주세요.")
                    st.error("YouTube API 요청이 금지되었습니다. API 키 권한을 확인해주세요.")
                    return []
                response.raise_for_status()
                data = response.json()
                
                video_ids = [item["id"]["videoId"] for item in data["items"]]
                
                # 영상 상세 정보 가져오기
                video_params = {
                    "part": "contentDetails,statistics,snippet",
                    "id": ",".join(video_ids),
                    "key": self.youtube_api_key
                }
                
                video_response = requests.get("https://www.googleapis.com/youtube/v3/videos", params=video_params)
                if video_response.status_code == 403:
                    logger.error("YouTube API 요청이 금지되었습니다. API 키 권한을 확인해주세요.")
                    st.error("YouTube API 요청이 금지되었습니다. API 키 권한을 확인해주세요.")
                    return []
                video_response.raise_for_status()  # API 응답 확인
                video_data = video_response.json()
                
                for item in video_data["items"]:
                    duration = self._parse_duration(item["contentDetails"]["duration"])
                    
                    if min_duration <= duration <= max_duration:
                        published_at = datetime.strptime(
                            item["snippet"]["publishedAt"],
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                        
                        video = VideoInfo(
                            title=item["snippet"]["title"],
                            url=f"https://www.youtube.com/watch?v={item['id']}",
                            duration=duration,
                            thumbnail_url=item["snippet"]["thumbnails"]["high"]["url"],
                            description=item["snippet"]["description"],
                            views=int(item["statistics"].get("viewCount", 0)),
                            published_at=published_at
                        )
                        videos.append(video)
                        
                        if len(videos) >= max_results:
                            break
                
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break
            
            return videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"YouTube API 요청 중 오류 발생: {str(e)}")
            st.error(f"YouTube API 요청 중 오류가 발생했습니다: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"비디오 검색 중 오류 발생: {str(e)}")
            st.error(f"비디오 검색 중 오류가 발생했습니다: {str(e)}")
            return []

    def get_video_transcript(self, video_url: str) -> str:
        """YouTube 영상의 자막을 추출"""
        try:
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'subtitleslangs': ['en'],
                'quiet': True,
                'outtmpl': '%(id)s.%(ext)s'
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if 'requested_subtitles' in info and info['requested_subtitles']:
                    sub_file = list(info['requested_subtitles'].values())[0]['url']
                    transcript_response = requests.get(sub_file)
                    transcript_response.raise_for_status()
                    transcript = transcript_response.text
                    return self._clean_transcript(transcript)
                else:
                    logger.warning(f"영상 '{info['title']}'에 대한 자동 생성 자막을 찾을 수 없습니다.")
                    return "자막을 찾을 수 없습니다."
        
        except Exception as e:
            logger.error(f"자막 추출 중 오류 발생: {str(e)}")
            return "자막 추출 중 오류가 발생했습니다."

    def generate_ideas(self, content: str, context: Optional[str] = None) -> str:
        """프로젝트 아이디어 생성"""
        try:
            context_text = f"추가 컨텍스트:\n{context}" if context else ""
        
            prompt = f"""다음은 YouTube 영상의 내용입니다:

{content}

{context_text}

이 내용을 바탕으로:
1. 실현 가능한 프로젝트 아이디어 3개를 제안해주세요.
2. 각 아이디어에 대해 다음 정보를 포함해주세요:
   - 프로젝트명
   - 주요 기능
   - 사용할 기술 스택
   - 예상 개발 기간
   - 난이도 (초급/중급/고급)
3. 각 아이디어의 장단점을 분석해주세요.
4. 추가 발전 방향을 제시해주세요."""
            
            session_id = "default_session"
            response = self.chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        except Exception as e:
            logger.error(f"아이디어 생성 중 오류 발생: {str(e)}")
            return "아이디어 생성 중 오류가 발생했습니다."

    def _parse_duration(self, duration_str: str) -> int:
        """ISO 8601 형식의 지속시간을 초로 변환"""
        try:
            duration = isodate.parse_duration(duration_str)
            return int(duration.total_seconds())
        except:
            return 0

    def _clean_transcript(self, transcript: str) -> str:
        """자막 텍스트 정리"""
        import re
        
        # HTML 태그 제거
        transcript = re.sub(r'<[^>]+>', '', transcript)
        # 중복 공백 제거
        transcript = re.sub(r'\s+', ' ', transcript)
        # 타임스탬프 제거
        transcript = re.sub(r'\[\d{2}:\d{2}\.\d{3}\]', '', transcript)
        
        return transcript.strip()

def main():
    st.set_page_config(
        page_title="YouTube 프로젝트 아이디어 생성기",
        page_icon=":star:",
        layout="wide"
    )

    # API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not openai_api_key or not youtube_api_key:
        st.error("OpenAI API 키와 YouTube API 키가 필요합니다. .env 파일을 확인해주세요.")
        return

    # 아이디어 생성기 초기화 - API 키 명시적 전달
    generator = YouTubeIdeaGenerator(
        openai_api_key=openai_api_key,
        youtube_api_key=youtube_api_key
    )

    # 스타일 적용
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .video-container {
                border: 1px solid #ddd;
                padding: 1rem;
                border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .idea-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌟 YouTube 프로젝트 아이디어 생성기")
    st.markdown("""
        YouTube 영상을 통해 프로젝트 아이디어를 얻고 발전시켜보세요.
        AI 멘토가 실현 가능한 프로젝트 아이디어를 제안해드립니다.
    """)

    # 사이드바 설정
    with st.sidebar:
        st.header("🔍 검색 설정")
        
        # 영상 길이 설정
        st.subheader("영상 길이")
        duration_range = st.slider(
            "영상 길이 범위(분)",
            min_value=0,
            max_value=60,
            value=(3, 15),
            step=1
        )
        
        # 정렬 옵션
        sort_by = st.selectbox(
            "정렬 기준",
            ["관련성", "조회수 ↓", "길이 ↓", "최신순 ↓"]
        )
        
        # 검색 결과 수
        max_results = st.slider("검색 결과 수", 1, 10, 1)
        
        st.markdown("---")
        st.markdown("""
            ### 💡 팁
            - 구체적인 기술이나 도메인으로 검색하세요
            - 실제 프로젝트 예시를 참고하세요
            - 최신 트렌드를 고려하세요
        """)

    # 메인 화면
    query = st.text_input(
        "관심 있는 주제나 기술을 검색하세요",
        placeholder="예: React Native 앱 개발, 아이디어 대회, 캡스톤 디자인, AI 챗봇, 데이터 분석"
    )

    if query:
        with st.spinner("영상 검색 중..."):
            videos = generator.search_videos(
                query=query,
                max_results=max_results,
                min_duration=duration_range[0] * 60,
                max_duration=duration_range[1] * 60
            )

            # 정렬 적용
            if sort_by == "조회수 ↓":
                videos.sort(key=lambda x: x.views, reverse=True)
            elif sort_by == "길이 ↓":
                videos.sort(key=lambda x: x.duration, reverse=True)
            elif sort_by == "최신순 ↓":
                videos.sort(key=lambda x: x.published_at, reverse=True)

        if videos:
            st.subheader("🎥 검색 결과")
            
            for idx, video in enumerate(videos):
                with st.expander(f"📺 {video.title} ({video.duration_str})"):
                    cols = st.columns([2, 1])
                    
                    with cols[0]:
                        st.image(video.thumbnail_url)
                    
                    with cols[1]:
                        st.markdown(f"**길이:** {video.duration_str}")
                        st.markdown(f"**조회수:** {video.views:,}회")
                        st.markdown(f"**게시일:** {video.published_at.strftime('%Y-%m-%d')}")
                    
                    st.markdown(f"**설명:** {video.description[:300]}...")
                    
                    # 고유한 key 값 사용
                    if st.button("이 영상으로 아이디어 얻기", key=f"select_video_{idx}"):
                        with st.spinner("영상 분석 중..."):
                            transcript = generator.get_video_transcript(video.url)
                            
                            if transcript:
                                # 기존의 nested expander를 제거하고 하나의 텍스트 영역으로 대체
                                st.text_area("영상 내용", transcript, height=200, key=f"transcript_{idx}")
                                
                                context = st.text_area(
                                    "추가 컨텍스트나 제약사항을 입력하세요 (선택사항)",
                                    placeholder="예: 초보자를 위한 프로젝트여야 함, Python만 사용, 2주 안에 완료 필요 등",
                                    key=f"context_{idx}"
                                )
                                
                                if st.button("아이디어 생성", key=f"generate_ideas_{idx}"):
                                    with st.spinner("아이디어 생성 중..."):                                             
                                        ideas = generator.generate_ideas(transcript, context)
                                        st.session_state[f"ideas_{idx}"] = ideas
                                        
                                        st.markdown("### 💡 생성된 아이디어")
                                        st.markdown(ideas)
                                        
                                        # 아이디어 저장 기능
                                        if "saved_ideas" not in st.session_state:
                                            st.session_state.saved_ideas = []
                                            
                                        if st.button("이 아이디어 저장", key=f"save_ideas_{idx}"):
                                            st.session_state.saved_ideas.append({
                                                "video_title": video.title,
                                                "ideas": ideas,
                                                "timestamp": datetime.now()
                                            })
                                            st.success("아이디어가 저장되었습니다!")
                                
                                # 아이디어 발전시키기
                                if f"ideas_{idx}" in st.session_state:
                                    st.markdown("### 🚀 아이디어 발전시키기")
                                    feedback = st.text_area(
                                        "아이디어에 대한 의견이나 질문을 입력하세요",
                                        key=f"feedback_{idx}",
                                        placeholder="예: 첫 번째 아이디어를 더 자세히 설명해주세요, 기술 스택을 단순화하고 싶어요 등"
                                    )
                                    
                                    if st.button("아이디어 발전시키기", key=f"refine_ideas_{idx}"):
                                        with st.spinner("아이디어 발전 중..."):
                                            refined = generator.refine_idea(
                                                st.session_state[f"ideas_{idx}"],
                                                feedback
                                            )
                                            st.markdown("### 🌱 발전된 아이디어")
                                            st.markdown(refined)
                            else:
                                st.error("자막을 추출할 수 없습니다. 다른 영상을 선택해주세요.")

    # 저장된 아이디어 표시
    if "saved_ideas" in st.session_state and st.session_state.saved_ideas:
        st.sidebar.markdown("---")
        st.sidebar.header("💾 저장된 아이디어")
        
        for idx, saved in enumerate(st.session_state.saved_ideas):
            with st.sidebar.expander(
                f"📌 {saved['video_title'][:30]}... ({saved['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            ):
                st.markdown(saved['ideas'])
                if st.button("삭제", key=f"delete_saved_idea_{idx}"):
                    st.session_state.saved_ideas.pop(idx)
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
