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
    GPT_MODEL = "gpt-4"  # GPT-4로 변경하여 더 높은 품질의 분석 가능
    MAX_TOKENS = 2000    # 응답 길이 제한
    TEMPERATURE = 0.7    # 창의성 조절 (0: 보수적, 1: 창의적)

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
        system_prompt = """당신은 콘텐츠 분석 전문가입니다. 
        주어진 텍스트에서 핵심 아이디어, 인사이트, 실행 가능한 조언을 추출하는 것이 당신의 임무입니다.
        
        다음 가이드라인을 따라 분석해주세요:
        1. 핵심 아이디어: 텍스트에서 가장 중요한 개념과 주장을 파악
        2. 실용적 인사이트: 실제 적용 가능한 통찰을 도출
        3. 액션 아이템: 구체적이고 실행 가능한 단계별 조언 제시
        4. 주요 데이터/통계: 언급된 중요한 수치나 데이터 포인트 정리
        5. 한계점/주의사항: 내용의 제한사항이나 고려해야 할 점 분석
        
        응답은 명확하고 구조화된 형태로 작성해주세요."""

        user_prompt = f"""다음 콘텐츠를 분석해주세요:

        {text}

        다음 형식으로 응답해주세요:

        📌 핵심 아이디어
        1.
        2.
        3.

        💡 주요 인사이트
        1.
        2.
        3.

        ⚡ 실행 가능한 액션 아이템
        1.
        2.
        3.

        📊 중요 데이터 포인트
        •
        •

        ⚠️ 고려사항 및 한계점
        •
        •

        💪 결론 및 권장사항
        
        """

        response = self.client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            top_p=0.9,             # 상위 90%의 확률을 가진 토큰만 고려
            frequency_penalty=0.6,  # 반복 줄이기
            presence_penalty=0.6    # 새로운 주제 도입 장려
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