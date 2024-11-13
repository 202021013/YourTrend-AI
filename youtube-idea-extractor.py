import yt_dlp
import whisper
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class Config:
    WHISPER_MODEL = "base"
    OUTPUT_DIR = "downloads"
    AUDIO_FORMAT = "mp3"

class YouTubeIdeaExtractor:
    def __init__(self):
        """
        Initialize the YouTube idea extractor using API key from environment variables
        """
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.model = whisper.load_model(Config.WHISPER_MODEL)
        
    def download_audio(self, youtube_url: str, output_dir: str = Config.OUTPUT_DIR) -> str:
        """
        Download YouTube video audio
        :param youtube_url: YouTube video URL
        :param output_dir: Directory to save the audio file
        :return: Path to the downloaded audio file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'paths': {'home': str(output_path)},
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': Config.AUDIO_FORMAT,
            }]
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + f".{Config.AUDIO_FORMAT}"
            return filename

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Whisper
        :param audio_path: Path to the audio file
        :return: Transcribed text
        """
        result = self.model.transcribe(audio_path)
        return result["text"]

    def extract_ideas(self, text: str) -> str:
        """
        Extract main ideas from the transcribed text using OpenAI API
        :param text: Transcribed text
        :return: List of extracted ideas
        """
        prompt = f"""아래 텍스트에서 주요 아이디어와 인사이트를 추출해주세요. 
        다음과 같은 형식으로 응답해주세요:
        1. [주요 아이디어 1]
        2. [주요 아이디어 2]
        3. [실행 가능한 인사이트]
        
        텍스트:
        {text}"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 콘텐츠에서 실용적인 아이디어와 인사이트를 추출하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content

    def process_video(self, youtube_url: str) -> Optional[str]:
        """
        Process YouTube video: download, transcribe, and extract ideas
        :param youtube_url: YouTube video URL
        :return: Extracted ideas
        """
        audio_path = None
        try:
            # 1. Download audio
            logger.info("다운로드 중...")
            audio_path = self.download_audio(youtube_url)
            
            # 2. Transcribe
            logger.info("음성을 텍스트로 변환 중...")
            transcribed_text = self.transcribe_audio(audio_path)
            
            # 3. Extract ideas
            logger.info("아이디어 추출 중...")
            ideas = self.extract_ideas(transcribed_text)
            
            return ideas
            
        except Exception as e:
            logger.error(f"에러 발생: {str(e)}")
            return None
        finally:
            # 4. Clean up
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    logger.error(f"파일 삭제 중 오류 발생: {e}")

# 사용 예시
if __name__ == "__main__":
    try:
        # 실제 YouTube 영상 URL을 입력하세요
        youtube_url = input("YouTube URL을 입력하세요: ")
        
        extractor = YouTubeIdeaExtractor()
        ideas = extractor.process_video(youtube_url)
        
        if ideas:
            logger.info("\n추출된 아이디어:")
            print(ideas)
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")