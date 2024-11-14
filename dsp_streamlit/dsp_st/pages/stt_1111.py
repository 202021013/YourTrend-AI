from google.cloud import speech_v1, storage
import os
from pydub import AudioSegment
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import yt_dlp
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import isodate
import pandas as pd
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
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
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes}:{seconds:02d}"
    
    def _format_duration(self) -> str:
        return self.duration_str

@dataclass
class TrendAnalysis:
    trending_topics: Dict[str, int]
    avg_duration: float
    popular_times: Dict[str, int]
    engagement_rate: float

class YouTubeTranscriptExtractor:
    def __init__(self):
        """Google Cloud 서비스 초기화"""
        self.llm_api_key = os.getenv('llm_api_key')
        self.speech_json_path = os.getenv('SPEECH_JSON_PATH')
        self.bucket_name = os.getenv('BUCKET_NAME')
        
        # Google Cloud 인증 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.speech_json_path
        self.speech_client = speech_v1.SpeechClient()
        self.storage_client = storage.Client()
        
    def download_and_convert_audio(self, video_url: str, audio_dir: str) -> Optional[str]:
        """YouTube 영상에서 오디오를 다운로드하고 변환"""
        try:
            # 디렉토리 존재 확인 및 생성
            os.makedirs(audio_dir, exist_ok=True)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_{timestamp}"
            converted_filename = f"converted_{timestamp}.wav"
            
            # 전체 경로 설정
            temp_path = os.path.join(audio_dir, temp_filename)
            converted_path = os.path.join(audio_dir, converted_filename)
            
            # yt-dlp 설정 수정
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True
            }
            
            # 다운로드 시도
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as e:
                logger.error(f"YouTube 다운로드 실패: {str(e)}")
                return None
                
            # 실제 생성된 파일 찾기
            wav_file = temp_path + '.wav'
            if not os.path.exists(wav_file):
                logger.error(f"다운로드된 파일을 찾을 수 없음: {wav_file}")
                return None
                
            # 오디오 변환
            try:
                audio = AudioSegment.from_wav(wav_file)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(converted_path, format='wav')
                
                # 임시 파일 정리
                try:
                    os.remove(wav_file)
                except OSError:
                    pass
                    
                return converted_path if os.path.exists(converted_path) else None
                
            except Exception as e:
                logger.error(f"오디오 변환 실패: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"오디오 처리 중 오류: {str(e)}")
            return None

    def transcribe_long_audio(self, audio_path: str, language_code: str = 'ko-KR') -> Optional[str]:
        """긴 오디오 파일의 음성을 텍스트로 변환"""
        try:
            # Cloud Storage에 파일 업로드
            bucket = self.storage_client.bucket(self.bucket_name)
            audio_blob_name = f"temp_audio_{os.path.basename(audio_path)}"
            blob = bucket.blob(audio_blob_name)
            
            # 오디오 파일 업로드
            blob.upload_from_filename(audio_path)
            gcs_uri = f"gs://{self.bucket_name}/{audio_blob_name}"
            
            # 오디오 설정
            audio = speech_v1.RecognitionAudio(uri=gcs_uri)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model='latest_long',
                use_enhanced=True
            )
            
            # 비동기 음성 인식 시작
            operation = self.speech_client.long_running_recognize(
                config=config,
                audio=audio
            )
            
            st.info("음성 인식이 진행 중입니다. 잠시만 기다려주세요...")
            
            # 결과 대기
            response = operation.result()
            
            # 임시 파일 삭제
            blob.delete()
            
            # 전체 텍스트 조합
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + "\n"
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"음성 인식 중 오류 발생: {str(e)}")
            return None
        finally:
            # 임시 파일이 남아있는 경우 삭제 시도
            try:
                blob = bucket.blob(audio_blob_name)
                if blob.exists():
                    blob.delete()
            except:
                pass

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

class EnhancedYouTubeIdeaGenerator:
    def __init__(self, openai_api_key: str, youtube_api_key: str):
        self.openai_api_key = openai_api_key
        self.youtube_api_key = youtube_api_key
        self.trend_cache = {}
        self.idea_history = []
        self.chain = ChatOpenAI(openai_api_key=openai_api_key)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def search_videos(self, query: str, max_results: int = 50) -> List[VideoInfo]:
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "key": self.youtube_api_key,
            "maxResults": max_results,
            "type": "video"
        }
        
        response = requests.get(url, params=params)
        videos = []
        
        for item in response.json().get("items", []):
            video_id = item["id"]["videoId"]
            stats_url = f"https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                "part": "statistics,contentDetails",
                "id": video_id,
                "key": self.youtube_api_key
            }
            
            stats_response = requests.get(stats_url, params=stats_params)
            stats_data = stats_response.json()["items"][0]
            
            duration = isodate.parse_duration(stats_data["contentDetails"]["duration"]).total_seconds()
            
            videos.append(VideoInfo(
                title=item["snippet"]["title"],
                url=f"https://www.youtube.com/watch?v={video_id}",
                duration=int(duration),
                thumbnail_url=item["snippet"]["thumbnails"]["high"]["url"],
                description=item["snippet"]["description"],
                views=int(stats_data["statistics"]["viewCount"]),
                published_at=datetime.strptime(item["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            ))
        
        return videos

    def analyze_trends(self, query: str, days: int = 7) -> TrendAnalysis:
        cache_key = f"{query}_{days}"
        if cache_key in self.trend_cache:
            return self.trend_cache[cache_key]
            
        videos = self.search_videos(query=query)
        
        all_text = " ".join([v.title + " " + v.description for v in videos])
        tokens = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        meaningful_words = [word for word in tokens if word.isalnum() and word not in stop_words]
        trending_topics = dict(Counter(meaningful_words).most_common(10))
        
        analysis = TrendAnalysis(
            trending_topics=trending_topics,
            avg_duration=sum(v.duration for v in videos) / len(videos) if videos else 0,
            popular_times=dict(Counter([v.published_at.hour for v in videos])),
            engagement_rate=len(videos) / sum(v.views for v in videos) if videos else 0
        )
        
        self.trend_cache[cache_key] = analysis
        return analysis

    def visualize_trends(self, trends: TrendAnalysis) -> Dict[str, px.Figure]:
        figures = {}
        
        keyword_df = pd.DataFrame(
            list(trends.trending_topics.items()),
            columns=['Keyword', 'Frequency']
        )
        figures['keywords'] = px.bar(
            keyword_df,
            x='Keyword',
            y='Frequency',
            title='트렌딩 키워드'
        )
        
        time_df = pd.DataFrame(
            list(trends.popular_times.items()),
            columns=['Hour', 'Uploads']
        )
        figures['upload_times'] = px.line(
            time_df,
            x='Hour',
            y='Uploads',
            title='인기 업로드 시간대'
        )
        
        return figures

def main():
    st.set_page_config(page_title="YouTube 프로젝트 아이디어 생성기", page_icon="🚀", layout="wide")
    
    st.title("🚀 YouTube 프로젝트 아이디어 생성기")
    st.markdown("""
        트렌드 분석과 AI를 활용한 스마트 프로젝트 아이디어 생성기
        * 실시간 트렌드 분석
        * 시장성 분석 포함
        * 데이터 기반 인사이트
    """)
    
    with st.sidebar:
        st.header("🎯 검색 & 분석 설정")
        query = st.text_input("검색어", placeholder="기술/도메인 입력")
        trend_days = st.slider("트렌드 분석 기간(일)", 1, 30, 7)
        
        st.markdown("---")
        st.markdown("### 📊 트렌드 필터")
        min_views = st.number_input("최소 조회수", 0, 1000000, 1000)
        engagement_threshold = st.slider("최소 참여율(%)", 0.0, 100.0, 1.0)
    
    if query:
        generator = EnhancedYouTubeIdeaGenerator(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY")
        )
        
        with st.spinner("트렌드 분석 중..."):
            trends = generator.analyze_trends(query, trend_days)
            figures = generator.visualize_trends(trends)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figures['keywords'])
            with col2:
                st.plotly_chart(figures['upload_times'])
            
            context = st.text_area("추가 컨텍스트", placeholder="특별한 요구사항이나 제약조건을 입력하세요")
            
            if st.button("아이디어 생성"):
                with st.spinner("AI가 아이디어를 생성중입니다..."):
                    prompt = generator.generate_prompt(query, trends, context)
                    response = generator.chain.invoke({"input": prompt})
                    
                    st.markdown("### 💡 생성된 아이디어")
                    st.markdown(response.content)

if __name__ == "__main__":
    main()