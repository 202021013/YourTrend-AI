import os
from datetime import datetime
import tempfile
import logging
from faster_whisper import WhisperModel
from dataclasses import dataclass
from typing import List, Dict, Optional
import yt_dlp
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class VideoInfo:
    title: str
    url: str
    duration: int
    thumbnail_url: str
    description: str
    view_count: int
    upload_date: datetime

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
        return " ".join(parts) if parts else "0초"

class YouTubeHandler:
    def __init__(self):
        self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
        self.ydl_base_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'noplaylist': True
        }

    def search_videos(self, query: str, max_results: int = 5, 
                     min_duration: int = 0, max_duration: int = 1200) -> List[VideoInfo]:
        ydl_opts = {
            **self.ydl_base_opts,
            'extract_flat': False,
            'format': 'best',
            'youtube_include_dash_manifest': False
        }

        videos = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 기본 검색 시도
            search_url = f"https://www.youtube.com/results?search_query={query}&sp=CAASBhABQgQIAQ%3D%3D"
            videos = self._process_search(ydl, search_url, max_results, min_duration, max_duration)

            # 백업 검색 시도
            if not videos:
                search_query = f"ytsearch{max_results}:{query}"
                videos = self._process_search(ydl, search_query, max_results, min_duration, max_duration)

        return videos

    def _process_search(self, ydl, search_query: str, max_results: int,
                       min_duration: int, max_duration: int) -> List[VideoInfo]:
        videos = []
        try:
            info = ydl.extract_info(search_query, download=False, process=False)
            if info and 'entries' in info:
                for entry in info['entries'][:max_results]:
                    try:
                        video_info = ydl.extract_info(entry['url'], download=False)
                        duration = video_info.get('duration', 0)
                        
                        if min_duration <= duration <= max_duration:
                            upload_date_str = video_info.get('upload_date', '')
                            upload_date = datetime.strptime(upload_date_str, '%Y%m%d') if upload_date_str else datetime.now()
                            
                            video = VideoInfo(
                                title=video_info.get('title', ''),
                                url=video_info.get('webpage_url', ''),
                                duration=duration,
                                thumbnail_url=video_info.get('thumbnail', ''),
                                description=video_info.get('description', ''),
                                view_count=video_info.get('view_count', 0),
                                upload_date=upload_date
                            )
                            videos.append(video)
                    except Exception as e:
                        logger.error(f"비디오 정보 추출 실패: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"검색 처리 실패: {str(e)}")
        return videos

    def extract_and_transcribe(self, video_url: str) -> Optional[str]:
        audio_path = self.extract_audio(video_url)
        if audio_path:
            transcript = self.transcribe_audio(audio_path)
            try:
                import os
                os.unlink(audio_path)
            except:
                pass
            return transcript
        return None

    def extract_audio(self, video_url: str) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                ydl_opts = {
                    **self.ydl_base_opts,
                    'format': 'bestaudio/best',
                    'outtmpl': temp_file.name,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }]
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                return temp_file.name
        except Exception as e:
            logger.error(f"오디오 추출 실패: {str(e)}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        try:
            segments, _ = self.whisper_model.transcribe(audio_path, language="ko")
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            logger.error(f"전사 실패: {str(e)}")
            return None

class IdeaManager:
    def __init__(self):
        if 'saved_ideas' not in st.session_state:
            st.session_state.saved_ideas = []

    def save_idea(self, video_title: str, ideas: str):
        st.session_state.saved_ideas.append({
            'video_title': video_title,
            'ideas': ideas,
            'timestamp': datetime.now()
        })

    def delete_idea(self, idx: int):
        st.session_state.saved_ideas.pop(idx)

class YouTubeIdeaGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.youtube_handler = YouTubeHandler()
        
        self.chat_model = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 창의적인 프로젝트 아이디어를 제안하는 AI 멘토입니다."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        self.chain = RunnableWithMessageHistory(
            self.prompt | self.chat_model,
            lambda session_id: self.memory,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        self.idea_manager = IdeaManager()

    def search_videos(self, query: str, max_results: int = 5, 
                     min_duration: int = 0, max_duration: int = 1200) -> List[VideoInfo]:
        return self.youtube_handler.search_videos(
            query=query,
            max_results=max_results,
            min_duration=min_duration,
            max_duration=max_duration
        )

    def get_video_transcript(self, video_url: str) -> Optional[str]:
        try:
            with st.spinner("오디오 처리를 시작합니다..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("오디오를 추출하는 중...")
                progress_bar.progress(25)
                
                transcript = self.youtube_handler.extract_and_transcribe(video_url)
                
                if transcript:
                    status_text.text("변환이 완료되었습니다!")
                    progress_bar.progress(100)
                    return transcript
                    
                return None
                
        except Exception as e:
            logger.error(f"전사 추출 실패: {str(e)}")
            st.error(f"전사 추출 실패: {str(e)}")
            return None

    def generate_ideas(self, content: str, context: Optional[str] = None) -> str:
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
            logger.error(f"아이디어 생성 실패: {str(e)}")
            return "아이디어 생성에 실패했습니다."

class VideoSearcher:
    def __init__(self):
        self.idea_manager = IdeaManager()  # IdeaManager 인스턴스 추가
        
    def search_videos(self, query: str, max_results: int = 5, 
                     min_duration: int = 0, max_duration: int = 1200) -> List[VideoInfo]:
        try:
            videos = []
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'format': 'best',
                'noplaylist': True,
                'youtube_include_dash_manifest': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = []
                search_query = f"ytsearch{max_results}:{query}"
                
                try:
                    results = ydl.extract_info(search_query, download=False)
                    if results and 'entries' in results:
                        search_results = list(results['entries'])
                        
                        for entry in search_results:
                            if not entry:
                                continue
                            
                            try:
                                video_info = ydl.extract_info(
                                    entry['id'], 
                                    download=False,
                                    process=True
                                )
                                
                                if not video_info:
                                    continue
                                    
                                duration = video_info.get('duration', 0)
                                if min_duration <= duration <= max_duration:
                                    video = self._create_video_info(video_info)
                                    videos.append(video)
                                    
                                    if len(videos) >= max_results:
                                        break
                                        
                            except Exception as e:
                                logger.error(f"비디오 정보 추출 실패: {str(e)}")
                                continue
                                
                except Exception as e:
                    logger.error(f"검색 처리 실패: {str(e)}")
            
            return videos
            
        except Exception as e:
            logger.error(f"동영상 검색 실패: {str(e)}")
            return []

    def _create_video_info(self, video_info: dict) -> VideoInfo:
        """비디오 정보 객체 생성 헬퍼 메소드"""
        upload_date_str = video_info.get('upload_date', '')
        upload_date = datetime.strptime(upload_date_str, '%Y%m%d') if upload_date_str else datetime.now()
        
        return VideoInfo(
            title=video_info.get('title', ''),
            url=f"https://www.youtube.com/watch?v={video_info.get('id', '')}",
            duration=video_info.get('duration', 0),
            thumbnail_url=video_info.get('thumbnail', ''),
            description=video_info.get('description', ''),
            view_count=video_info.get('view_count', 0),
            upload_date=upload_date
        )

    def display_search_results(self, videos: List[VideoInfo]):
        """검색 결과 표시 및 아이디어 저장 처리"""
        if videos:
            for idx, video in enumerate(videos):
                with st.expander(f"📺 {video.title}"):
                    st.image(video.thumbnail_url, use_column_width=True)
                    transcript = self._get_transcript(video.url)
                    
                    if transcript:
                        ideas = self._generate_ideas(transcript)
                        st.markdown(ideas)
                        
                        if st.button("아이디어 저장하기", key=f"save_ideas_{idx}"):
                            self.idea_manager.save_idea(video.title, ideas)
                            st.success("아이디어가 저장되었습니다!")
                    else:
                        st.error("전사를 추출할 수 없습니다. 다른 영상을 시도해주세요.")
        else:
            st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")

    def display_saved_ideas(self):
        """저장된 아이디어 사이드바 표시"""
        if st.session_state.get('saved_ideas'):
            st.sidebar.markdown("---")
            st.sidebar.header("💾 저장된 아이디어")
            
            for idx, saved in enumerate(st.session_state.saved_ideas):
                with st.sidebar.expander(
                    f"📌 {saved['video_title'][:30]}... ({saved['timestamp'].strftime('%Y-%m-%d %H:%M')})"
                ):
                    st.markdown(saved['ideas'])
                    if st.button("삭제", key=f"delete_saved_idea_{idx}"):
                        self.idea_manager.delete_idea(idx)
                        st.experimental_rerun()

def main():
    searcher = VideoSearcher()
    query = st.text_input("YouTube 검색어를 입력하세요")
    
    if query:
        videos = searcher.search_videos(query)
        searcher.display_search_results(videos)
    
    searcher.display_saved_ideas()

if __name__ == "__main__":
    main()