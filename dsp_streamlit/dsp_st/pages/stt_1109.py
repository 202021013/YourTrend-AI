from google.cloud import speech_v1
from google.cloud import storage
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
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import isodate

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
        return self._format_duration()
    
    def _format_duration(self) -> str:
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

class YouTubeIdeaGenerator:
    def __init__(self, openai_api_key: str, youtube_api_key: str):
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        self.transcript_extractor = YouTubeTranscriptExtractor()
        
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
        self.idea_manager = IdeaManager()

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

    def get_video_transcript(self, video_url: str) -> Optional[str]:
        try:
            with st.spinner("음성 인식을 시작합니다..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 현재 작업 디렉토리 기반으로 audio 폴더 생성
                base_dir = os.path.dirname(os.path.abspath(__file__))
                audio_dir = os.path.join(base_dir, 'audio')
                os.makedirs(audio_dir, exist_ok=True)
                
                status_text.text("오디오 추출 중...")
                progress_bar.progress(25)
                
                # 오디오 다운로드 및 변환
                converted_path = self.transcript_extractor.download_and_convert_audio(
                    video_url=video_url,
                    audio_dir=audio_dir
                )
                
                if not converted_path:
                    return None
                    
                status_text.text("음성 인식 처리 중...")
                progress_bar.progress(50)
                
                transcript = self.transcript_extractor.transcribe_long_audio(
                    audio_path=converted_path,
                    language_code='ko-KR'
                )
                
                if transcript:
                    status_text.text("음성 인식 완료!")
                    progress_bar.progress(100)
                    return transcript
                    
                return None
                
        except Exception as e:
            logger.exception(f"트랜스크립트 추출 중 오류 발생: {str(e)}")
            st.error(f"자막 추출 실패: {str(e)}")
            return None

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
    # 페이지 설정
    st.set_page_config(
        page_title="YouTube 프로젝트 아이디어 생성기",
        page_icon=":star:",
        layout="wide"
    )

    # 세션 상태 초기화
    if 'session_state' not in st.session_state:
        st.session_state.update({
            'generated_ideas': {},
            'selected_video': None,
            'transcripts': {},
            'search_results': [],
            'sort_option': '관련성'
        })

    # API 키 유효성 검사
    api_keys = validate_api_keys()
    if not api_keys:
        return

    # 아이디어 생성기 초기화 
    generator = initialize_generator(api_keys)
    if not generator:
        return

    # 사이드바 설정
    search_params = setup_sidebar()

    # 메인 검색 UI
    query = st.text_input(
        "관심 있는 주제나 기술을 검색하세요",
        placeholder="예: React Native 앱 개발, AI 챗봇, 데이터 분석"
    )

    if query:
        handle_search(query, generator, search_params)

    # 아이디어 표시 및 관리
    display_saved_ideas(generator)

def validate_api_keys():
    """API 키 유효성 검사"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not openai_api_key or not youtube_api_key:
        st.error("API 키가 없습니다. .env 파일을 확인해주세요.")
        return None
    
    return {'openai': openai_api_key, 'youtube': youtube_api_key}

def initialize_generator(api_keys):
    """아이디어 생성기 초기화"""
    try:
        return YouTubeIdeaGenerator(
            openai_api_key=api_keys['openai'],
            youtube_api_key=api_keys['youtube']
        )
    except Exception as e:
        st.error(f"생성기 초기화 실패: {str(e)}")
        return None

def setup_sidebar():
    """사이드바 설정"""
    with st.sidebar:
        st.header("🔍 검색 설정")
        
        duration_range = st.slider(
            "영상 길이 범위(분)",
            0, 60, (3, 15)
        )
        
        sort_by = st.selectbox(
            "정렬 기준",
            ["관련성", "조회수 ↓", "길이 ↓", "최신순 ↓"]
        )
        
        max_results = st.slider(
            "검색 결과 수", 
            1, 10, 5
        )
        
        display_search_tips()
        
        return {
            'duration': duration_range,
            'sort': sort_by,
            'max_results': max_results
        }

def handle_search(query, generator, params):
    """검색 실행 및 결과 처리"""
    with st.spinner("검색 중..."):
        videos = generator.search_videos(
            query=query,
            max_results=params['max_results'],
            min_duration=params['duration'][0] * 60,
            max_duration=params['duration'][1] * 60
        )
        
        if videos:
            display_videos(videos, generator)
        else:
            st.warning("검색 결과가 없습니다.")

def display_videos(videos, generator):
    """비디오 목록 표시"""
    st.subheader("🎥 검색 결과")
    
    for idx, video in enumerate(videos):
        with st.container():
            display_video_card(idx, video, generator)

def handle_idea_generation(video, transcript, context, idx, generator):
    """아이디어 생성 처리"""
    with st.spinner("아이디어 생성 중..."):
        ideas = generator.generate_ideas(transcript, context)
        st.session_state.generated_ideas[idx] = ideas
        
        st.markdown("### 💡 생성된 아이디어")
        st.markdown(ideas)
        
        if st.button("아이디어 저장", key=f"save_{idx}"):
            generator.idea_manager.save_idea(video.title, ideas)
            st.success("저장 완료!")

# 나머지 유틸리티 함수들...

if __name__ == "__main__":
    main()