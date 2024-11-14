import os
from datetime import datetime
import tempfile
import logging
import whisper
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
        return " ".join(parts)

class YouTubeTranscriptExtractor:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
    
    def extract_audio(self, video_url: str) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': temp_file.name,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'quiet': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                return temp_file.name
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None
        finally:
            try:
                os.remove(audio_path)
            except:
                pass

class VideoSearcher:
    def search_videos(self, query: str, max_results: int = 5, 
                     min_duration: int = 0, max_duration: int = 1200) -> List[VideoInfo]:
        try:
            videos = []
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'force_generic_extractor': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_query = f"ytsearch{max_results}:{query}"
                results = ydl.extract_info(search_query, download=False)
                
                for entry in results['entries']:
                    if not entry:
                        continue
                        
                    duration = entry.get('duration', 0)
                    if min_duration <= duration <= max_duration:
                        upload_date_str = entry.get('upload_date', '')
                        upload_date = datetime.strptime(upload_date_str, '%Y%m%d') if upload_date_str else datetime.now()
                        
                        video = VideoInfo(
                            title=entry.get('title', ''),
                            url=entry.get('url', ''),
                            duration=duration,
                            thumbnail_url=entry.get('thumbnail', ''),
                            description=entry.get('description', ''),
                            view_count=entry.get('view_count', 0),
                            upload_date=upload_date
                        )
                        videos.append(video)
                        
                        if len(videos) >= max_results:
                            break
            
            return videos
            
        except Exception as e:
            logger.error(f"Video search failed: {str(e)}")
            return []

class IdeaManager:
    def __init__(self):
        if 'saved_ideas' not in st.session_state:
            st.session_state.saved_ideas = []
    
    def save_idea(self, video_title: str, ideas: str) -> None:
        st.session_state.saved_ideas.append({
            "video_title": video_title,
            "ideas": ideas,
            "timestamp": datetime.now()
        })
    
    def delete_idea(self, index: int) -> None:
        if 0 <= index < len(st.session_state.saved_ideas):
            st.session_state.saved_ideas.pop(index)
            
    def get_all_ideas(self) -> List[Dict]:
        return st.session_state.saved_ideas

class YouTubeIdeaGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.transcript_extractor = YouTubeTranscriptExtractor()
        self.video_searcher = VideoSearcher()
        
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
        return self.video_searcher.search_videos(
            query=query,
            max_results=max_results,
            min_duration=min_duration,
            max_duration=max_duration
        )

    def get_video_transcript(self, video_url: str) -> Optional[str]:
        try:
            with st.spinner("Starting audio processing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting audio...")
                progress_bar.progress(25)
                
                audio_path = self.transcript_extractor.extract_audio(video_url)
                if not audio_path:
                    return None
                    
                status_text.text("Transcribing audio...")
                progress_bar.progress(50)
                
                transcript = self.transcript_extractor.transcribe_audio(audio_path)
                
                if transcript:
                    status_text.text("Transcription complete!")
                    progress_bar.progress(100)
                    return transcript
                    
                return None
                
        except Exception as e:
            logger.error(f"Transcript extraction failed: {str(e)}")
            st.error(f"Failed to extract transcript: {str(e)}")
            return None

    def generate_ideas(self, content: str, context: Optional[str] = None) -> str:
        try:
            context_text = f"Additional context:\n{context}" if context else ""
            
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
            logger.error(f"Idea generation failed: {str(e)}")
            return "Failed to generate ideas."

def main():
    st.set_page_config(
        page_title="YouTube Project Idea Generator",
        page_icon=":star:",
        layout="wide"
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key is required. Please check your .env file.")
        return

    generator = YouTubeIdeaGenerator(openai_api_key=openai_api_key)

    st.title("🌟 YouTube Project Idea Generator")
    
    with st.sidebar:
        st.header("🔍 Search Settings")
        duration_range = st.slider(
            "Video length (minutes)",
            min_value=0,
            max_value=60,
            value=(3, 4),
            step=1
        )
        
        sort_by = st.selectbox(
            "Sort by",
            ["Relevance", "Views ↓", "Length ↓", "Latest ↓"]
        )
        
        max_results = st.slider("Number of results", 1, 10, 1)

    query = st.text_input(
        "Search for topics or technologies",
        placeholder="e.g., React Native app development, AI chatbot, Data analysis"
    )

    if query:
        with st.spinner("Searching videos..."):
            videos = generator.search_videos(
                query=query,
                max_results=max_results,
                min_duration=duration_range[0] * 60,
                max_duration=duration_range[1] * 60
            )

            if sort_by == "Views ↓":
                videos.sort(key=lambda x: x.view_count, reverse=True)
            elif sort_by == "Length ↓":
                videos.sort(key=lambda x: x.duration, reverse=True)
            elif sort_by == "Latest ↓":
                videos.sort(key=lambda x: x.upload_date, reverse=True)

        if videos:
            st.subheader("🎥 Search Results")
            
            for idx, video in enumerate(videos):
                st.markdown("---")
                cols = st.columns([2, 1])
                
                with cols[0]:
                    st.image(video.thumbnail_url)
                    st.markdown(f"### 📺 {video.title}")
                
                with cols[1]:
                    st.markdown(f"**Length:** {video.duration_str}")
                    st.markdown(f"**Views:** {video.view_count:,}")
                    st.markdown(f"**Upload date:** {video.upload_date.strftime('%Y-%m-%d')}")
                    st.markdown(f"**Description:** {video.description[:300]}...")
                
                if st.button("Get ideas from this video", key=f"select_video_{idx}"):
                    transcript = generator.get_video_transcript(video.url)
                    
                    if transcript:
                        st.markdown("### 📝 Video Content")
                        st.text_area("Transcript", transcript, height=200, key=f"transcript_{idx}")
                        
                        context = st.text_area(
                            "Additional context or constraints (optional)",
                            key=f"context_{idx}"
                        )
                        
                        if st.button("Generate Ideas", key=f"generate_ideas_{idx}"):
                            with st.spinner("Generating ideas..."):
                                ideas = generator.generate_ideas(transcript, context)
                                st.session_state[f"ideas_{idx}"] = ideas
                                
                                st.markdown("### 💡 Generated Ideas")
                                st.markdown(ideas)
                                
                                if st.button("Save Ideas", key=f"save_ideas_{idx}"):
                                    generator.idea_manager.save_idea(video.title, ideas)
                                    st.success("Ideas saved!")
                    else:
                        st.error("Could not extract transcript. Please try another video.")

    if st.session_state.get('saved_ideas'):
        st.sidebar.markdown("---")
        st.sidebar.header("💾 Saved Ideas")
        
        for idx, saved in enumerate(st.session_state.saved_ideas):
            with st.sidebar.expander(
                f"📌 {saved['video_title'][:30]}... ({saved['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            ):
                st.markdown(saved['ideas'])
                if st.button("Delete", key=f"delete_saved_idea_{idx}"):
                    generator.idea_manager.delete_idea(idx)
                    st.experimental_rerun()

if __name__ == "__main__":
    main()