import streamlit as st
import yt_dlp
import whisper
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Optional
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()


class Config:
    WHISPER_MODEL = "base"
    OUTPUT_DIR = "downloads"
    AUDIO_FORMAT = "mp3"
    OPENAI_MODEL = "gpt-4o-mini"
    MAX_TEXT_LENGTH = 4000


class ConversationManager:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        persist_directory = "chroma_db"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="conversations"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_conversation(self, text: str, metadata: dict = None):
        texts = self.text_splitter.split_text(text)
        if texts:  # 텍스트가 있는 경우에만 추가
            self.vector_store.add_texts(
                texts,
                metadatas=[metadata] * len(texts) if metadata else None
            )

    def add_conversation(self, text: str, metadata: dict = None):
        try:
            texts = self.text_splitter.split_text(text)
            if texts:  # 텍스트가 있는 경우에만 추가
                self.vector_store.add_texts(
                    texts,
                    metadatas=[metadata] * len(texts) if metadata else None
                )
                self.vector_store.persist()  # 변경사항 저장
        except Exception as e:
            logger.error(f"대화 저장 중 오류 발생: {str(e)}")

    def search_similar(self, query: str, k: int = 3):
        return self.vector_store.similarity_search(query, k=k)


class YouTubeIdeaExtractor:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found in environment variables")

        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.model = whisper.load_model(Config.WHISPER_MODEL)
        self.conversation_manager = ConversationManager(self.OPENAI_API_KEY)

    def search_videos_with_pagination(self, query: str, page: int = 1, per_page: int = 5) -> list:
        """
        페이지네이션이 적용된 YouTube 영상 검색
        """
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'force_generic_extractor': False,
                'no_warnings': True,
                'playlistend': per_page * page  # 전체 결과 수 증가
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_url = f"ytsearch{per_page * page}:{query}"
                results = ydl.extract_info(search_url, download=False)

                videos = []
                if 'entries' in results:
                    # 현재 페이지에 해당하는 결과만 반환
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    entries = results['entries'][start_idx:end_idx]

                    for video in entries:
                        video_id = video.get('id', '')
                        videos.append({
                            'title': video.get('title', '제목 없음'),
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'duration': str(video.get('duration', 0)),
                            'channel': video.get('uploader', '채널명 없음'),
                            'description': video.get('description', '설명 없음'),
                            'thumbnail': f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                            'video_id': video_id
                        })
                return videos

        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return []

    def download_and_process(self, youtube_url: str) -> str:
        output_path = Path(Config.OUTPUT_DIR)
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
            return ydl.prepare_filename(info).rsplit(".", 1)[0] + f".{Config.AUDIO_FORMAT}"

    def transcribe_audio(self, audio_path: str) -> str:
        return self.model.transcribe(audio_path)["text"]

    def analyze_with_summary(self, text: str, summary: str, similar_contexts: list) -> str:
        """요약본과 이전 대화 기록을 바탕으로 심층 분석"""
        try:
            context = "\n".join([doc.page_content for doc in similar_contexts])

            prompt = f"""현재 콘텐츠와 이전 대화 기록을 바탕으로 심층 분석을 진행해주세요.

    현재 콘텐츠 요약:
    {summary}

    이전 대화 컨텍스트:
    {context}

    현재 텍스트 전문:
    {text[:2000]}...

    다음 항목들을 분석해주세요:
    1. 핵심 인사이트 (3개)
    2. 이전 내용과의 연관성 및 차이점
    3. 실행 가능한 제안사항 (2-3개)
    4. 추가 탐구가 필요한 부분"""

            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 전문적인 콘텐츠 분석가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"분석 생성 중 오류 발생: {str(e)}")
            return None

    def extract_ideas_with_context(self, text: str, similar_contexts: list) -> str:
        context = "\n".join([doc.page_content for doc in similar_contexts])

        if len(text) > Config.MAX_TEXT_LENGTH:
            text = text[:Config.MAX_TEXT_LENGTH] + "..."

        prompt = f"""현재 텍스트와 이전 컨텍스트를 기반으로 분석해주세요.

이전 컨텍스트:
{context}

현재 텍스트:
{text}

다음 형식으로 응답해주세요:
1. 주요 아이디어 (3개)
2. 실행 가능한 인사이트 (2개)
3. 이전 컨텍스트와의 연관성"""

        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 콘텐츠 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"아이디어 추출 중 오류 발생: {str(e)}")
            return None

    def summarize_transcript(self, text: str) -> str:
        """
        전사된 텍스트의 요약본 생성
        """
        try:
            prompt = f"""다음 텍스트의 주요 내용을 요약해주세요:

    텍스트:
    {text[:4000]}  # 토큰 제한을 고려하여 잘라냄

    요약 형식:
    1. 핵심 주제
    2. 주요 논점 (3-4개)
    3. 결론
    """
            response = self.client.chat.completions.create(
                model="gpt-4-1106-preview",  # 최신 모델 사용
                messages=[
                    {"role": "system", "content": "당신은 전문적인 콘텐츠 요약 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3  # 더 일관된 요약을 위해 낮은 temperature 사용
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"요약 생성 중 오류 발생: {str(e)}")
            return None

    def generate_analysis_with_history(self, current_text: str, summary: str) -> str:
        """
        요약본과 이전 대화 기록을 바탕으로 심층 분석 생성
        """
        try:
            # 이전 대화 컨텍스트 검색
            similar_contexts = self.conversation_manager.search_similar(
                current_text)
            context = "\n".join([doc.page_content for doc in similar_contexts])

            prompt = f"""현재 콘텐츠와 이전 대화 기록을 바탕으로 심층 분석을 진행해주세요.

    현재 콘텐츠 요약:
    {summary}

    이전 대화 컨텍스트:
    {context}

    다음 항목들을 분석해주세요:
    1. 핵심 인사이트 (3개)
    2. 이전 내용과의 연관성 및 차이점
    3. 실행 가능한 제안사항 (2-3개)
    4. 추가 탐구가 필요한 부분
    """
            response = self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "당신은 전문적인 콘텐츠 분석가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"분석 생성 중 오류 발생: {str(e)}")
            return None

    def generate_report(self, team_member: str, report_title: str, analysis: str) -> str:
        """
        분석 결과를 바탕으로 공식 보고서 생성
        """
        try:
            current_date = datetime.now().strftime("%Y년 %m월 %d일")
            prompt = f"""다음 분석 결과를 바탕으로 공식 보고서를 작성해주세요.

    기본 정보:
    - 작성자: {team_member}
    - 보고서 제목: {report_title}
    - 작성일: {current_date}

    분석 내용:
    {analysis}

    다음 형식으로 작성해주세요:
    1. 보고서 헤더 (제목, 작성자, 날짜)
    2. 요약 (3-4줄)
    3. 주요 발견사항 (3-4개)
    4. 세부 분석 내용
    5. 결론 및 제안사항
    6. 향후 연구 방향

    보고서는 학술적이고 전문적인 톤으로 작성해주세요."""

            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 전문적인 보고서 작성 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return None

    def process_video(self, youtube_url: str) -> Optional[tuple[str, str, str]]:
        """
        영상 처리 및 분석 (요약본과 심층 분석 포함)
        """
        audio_path = None
        try:
            logger.info("다운로드 중...")
            audio_path = self.download_and_process(youtube_url)

            logger.info("음성을 텍스트로 변환 중...")
            transcribed_text = self.transcribe_audio(audio_path)

            logger.info("텍스트 요약 중...")
            summary = self.summarize_transcript(transcribed_text)

            logger.info("심층 분석 중...")
            analysis = self.generate_analysis_with_history(
                transcribed_text, summary)

            # 대화 내용 저장
            self.conversation_manager.add_conversation(
                transcribed_text,
                metadata={
                    "url": youtube_url,
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary,
                    "analysis": analysis
                }
            )

            return transcribed_text, summary, analysis

        except Exception as e:
            logger.error(f"에러 발생: {str(e)}")
            return None, None, None
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    logger.error(f"파일 삭제 중 오류 발생: {e}")


def create_streamlit_app():
    st.set_page_config(
        page_title="YouTube 영상 분석기",
        page_icon="🎥",
        layout="wide"
    )

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("환경 변수에서 OpenAI API 키를 찾을 수 없습니다.")
        return

    st.title("🎥 YouTube 영상 분석기")
    st.markdown("YouTube 영상의 내용을 분석하고 주요 아이디어를 추출합니다.")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'extractor' not in st.session_state:
        st.session_state.extractor = YouTubeIdeaExtractor()

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        st.success("API 키가 로드되었습니다.")

        if st.button("🗑 기록 삭제"):
            st.session_state.history = []
            st.success("분석 기록이 삭제되었습니다.")

    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])

    with col1:
        search_query = st.text_input(
            "🔍 분석하고 싶은 주제를 입력하세요",
            placeholder="예: AI 챗봇 개발, 파이썬 프로그래밍 등"
        )

    if search_query:
        # 새 검색어가 입력되면 페이지 리셋
        if 'previous_query' not in st.session_state or st.session_state.previous_query != search_query:
            st.session_state.page = 1
            st.session_state.previous_query = search_query

        # 페이지 상태 관리
        if 'page' not in st.session_state:
            st.session_state.page = 1

        with st.spinner("영상 검색 중..."):
            try:
                videos = st.session_state.extractor.search_videos_with_pagination(
                    search_query,
                    page=st.session_state.page
                )

                if videos:
                    st.markdown("### 🎬 검색된 영상")

                    for idx, video in enumerate(videos):
                        st.markdown("---")
                        col_thumb, col_info = st.columns([1, 2])

                        with col_thumb:
                            st.image(video['thumbnail'])

                        with col_info:
                            st.markdown(f"### {video['title']}")
                            st.markdown(f"**채널**: {video['channel']}")

                            try:
                                duration = float(video['duration'])
                                minutes = int(duration // 60)
                                seconds = int(duration % 60)
                                st.markdown(f"**길이**: {minutes}분 {seconds}초")
                            except (ValueError, TypeError):
                                st.markdown("**길이**: 정보 없음")

                            description = video.get('description', '설명 없음')
                            if description and len(description) > 200:
                                description = description[:200] + "..."
                            st.markdown(f"**설명**: {description}")

                            # 비디오 정보 저장을 위한 세션 상태 추가
                            if 'video_data' not in st.session_state:
                                st.session_state.video_data = {}

                            # 비디오 분석 버튼 클릭 시
                            if st.button("🎯 이 영상 분석하기", key=f"analyze_{idx}"):
                                try:
                                    # 현재 비디오 정보 저장
                                    current_video_id = video['video_id']
                                    st.session_state.video_data[idx] = {
                                        'video_id': current_video_id,
                                        'url': video['url']
                                    }
                                    
                                    with st.spinner("🎵 영상 다운로드 중..."):
                                        # 올바른 비디오 URL 사용 확인
                                        video_url = st.session_state.video_data[idx]['url']
                                        audio_path = st.session_state.extractor.download_and_process(video_url)

                                    with st.spinner("🎯 음성을 텍스트로 변환 중..."):
                                        transcribed_text = st.session_state.extractor.transcribe_audio(
                                            audio_path)
                                        st.text_area(
                                            "📝 변환된 텍스트",
                                            transcribed_text,
                                            height=200
                                        )

                                    with st.spinner("📋 텍스트 요약 중..."):
                                        summary = st.session_state.extractor.summarize_transcript(
                                            transcribed_text)
                                        if summary:
                                            st.markdown("### 📌 요약")
                                            st.markdown(summary)

                                    with st.spinner("💡 심층 분석 중..."):
                                        similar_contexts = st.session_state.extractor.conversation_manager.search_similar(
                                            transcribed_text)
                                        analysis = st.session_state.extractor.analyze_with_summary(
                                            transcribed_text,
                                            summary,
                                            similar_contexts
                                        )

                                        # 분석 결과를 session_state에 저장
                                        if 'current_analysis' not in st.session_state:
                                            st.session_state.current_analysis = {}

                                        # 보고서 생성 섹션
                                        if analysis:
                                            st.markdown("### 📑 보고서 생성")
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                team_member = st.text_input("작성자 이름", key=f"team_{idx}")
                                            with col2:
                                                report_title = st.text_input("보고서 제목", key=f"title_{idx}")
                                            
                                            if st.button("보고서 생성", key=f"report_{idx}"):
                                                if team_member and report_title:
                                                    with st.spinner("보고서 생성 중..."):
                                                        try:
                                                            report = st.session_state.extractor.generate_report(
                                                                team_member,
                                                                report_title,
                                                                analysis
                                                            )
                                                            
                                                            if report:
                                                                # 보고서를 마크다운 파일로 저장
                                                                report_filename = f"report_{team_member}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                                                                with open(report_filename, "w", encoding="utf-8") as f:
                                                                    f.write(report)
                                                                
                                                                # 다운로드 버튼 생성
                                                                with open(report_filename, "rb") as f:
                                                                    st.download_button(
                                                                        label="📥 보고서 다운로드",
                                                                        data=f,
                                                                        file_name=report_filename,
                                                                        mime="text/markdown"
                                                                    )
                                                                
                                                                # 화면에도 표시
                                                                st.markdown("### 📊 생성된 보고서")
                                                                st.markdown(report)
                                                                st.success("보고서가 생성되었습니다!")
                                                                
                                                        except Exception as e:
                                                            st.error(f"보고서 생성 중 오류가 발생했습니다: {str(e)}")
                                                else:
                                                    st.warning("작성자 이름과 보고서 제목을 모두 입력해주세요.")
        
                                    if os.path.exists(audio_path):
                                        os.remove(audio_path)

                                except Exception as e:
                                    st.error(f"오류가 발생했습니다: {str(e)}")

                    # 페이지네이션 컨트롤
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        if st.session_state.page > 1:
                            if st.button("◀ 이전", key="prev_page"):
                                st.session_state.page -= 1
                                st.rerun()
                    with col2:
                        st.markdown(
                            f"<div style='text-align: center'>**페이지 {st.session_state.page}**</div>", unsafe_allow_html=True)
                    with col3:
                        if len(videos) == 5:
                            if st.button("다음 ▶", key="next_page"):
                                st.session_state.page += 1
                                st.rerun()

                else:
                    st.warning("검색 결과가 없습니다.")

            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")

    # 분석 기록
    with col2:
        st.markdown("### 📚 분석 기록")

        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"분석 #{len(st.session_state.history) - idx}"):
                if item.get('thumbnail'):
                    st.image(item['thumbnail'])
                st.markdown(f"**검색어**: {item['query']}")
                st.markdown(f"**제목**: {item['title']}")
                st.markdown(f"**시간**: {item['timestamp']}")

                tab1, tab2, tab3 = st.tabs(["요약", "분석", "전체 텍스트"])
                with tab1:
                    if item.get('summary'):
                        st.markdown(item['summary'])
                with tab2:
                    if item.get('analysis'):  # 'ideas' 대신 'analysis' 사용
                        st.markdown(item['analysis'])
                with tab3:
                    if item.get('text'):
                        st.text_area("원본 텍스트", item['text'], height=200)

                # 보고서가 있는 경우 표시
                if item.get('report'):
                    with st.expander("📑 생성된 보고서"):
                        st.markdown(item['report'])


if __name__ == "__main__":
    create_streamlit_app()
    