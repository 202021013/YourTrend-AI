import os
import re
import requests
import yt_dlp
import streamlit as st
from google.cloud import storage, speech
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

gcs_json_path = os.getenv("GCS_JSON_PATH")
speech_json_path = os.getenv("SPEECH_JSON_PATH")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
bucket_name = os.getenv("BUCKET_NAME")

# Streamlit 페이지 설정
st.set_page_config(page_title="YouTube Audio Extractor", layout="wide")
st.title("YouTube Audio Extractor & Project Idea Generator")

# 사용자로부터 검색어 입력받기
query = st.text_input("검색어를 입력하세요:")
num_videos_to_extract = st.slider("추출할 비디오 개수를 선택하세요 (최대 6개)", 1, 6, 3)

# OpenAI 및 Google 클라우드 클라이언트 설정
client = OpenAI(api_key=openai_api_key)

if not os.path.exists(gcs_json_path):
    st.error(f"GCS JSON 파일이 존재하지 않습니다: {gcs_json_path}")
else:
    storage_client = storage.Client.from_service_account_json(gcs_json_path)
    speech_client = speech.SpeechClient.from_service_account_json(speech_json_path)

def get_top_videos(query, max_results=10, page_token=None):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&maxResults={max_results}&type=video&key={youtube_api_key}"
    if page_token:
        url += f"&pageToken={page_token}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    videos = data.get("items", [])
    return [
        {
            "title": video["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={video['id']['videoId']}"
        }
        for video in videos if "videoId" in video["id"]
    ]

def get_video_duration(video_url):
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict.get("duration", 0)

def sanitize_title(title):
    return re.sub(r'[<>:"/\\|?*#]', '', title).strip()

def extract_audio(video_url, title):
    sanitized_title = sanitize_title(title)
    output_path = os.path.join(os.getcwd(), "audio_files", sanitized_title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return f"{output_path}.mp3"
    except Exception as e:
        st.error(f"오디오 추출 오류: {e}")
        return None

def upload_to_gcs(file_path):
    try:
        blob = storage_client.bucket(bucket_name).blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{os.path.basename(file_path)}"
    except Exception as e:
        st.error(f"GCS 업로드 오류: {e}")
        return None

def transcribe_audio(gcs_uri):
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)

    transcripts = [result.alternatives[0].transcript for result in response.results]
    return "\n".join(transcripts)

# 검색어가 있는 경우 비디오 추출 진행
if query:
    if st.button("비디오 검색 및 처리 시작"):
        videos = get_top_videos(query, max_results=num_videos_to_extract)
        transcriptions = []
        file_counter = 1

        for video in videos:
            duration = get_video_duration(video['url'])
            if duration <= 120:
                st.write(f"비디오 오디오 추출 중: {video['title']}")
                audio_file = extract_audio(video['url'], video['title'])
                if audio_file:
                    gcs_uri = upload_to_gcs(audio_file)
                    if gcs_uri:
                        text = transcribe_audio(gcs_uri)
                        transcriptions.append(text)
                        st.write(f"전사 완료: {video['title']}")
                        file_counter += 1
                    os.remove(audio_file)
                    if file_counter > num_videos_to_extract:
                        break
                else:
                    st.warning(f"오디오 추출 실패: {video['title']}")

        if transcriptions:
            st.write("모든 전사 결과:")
            complete_transcription = "\n".join(transcriptions)
            st.text_area("전사 내용", complete_transcription, height=200)

            # 프로젝트 아이디어 생성
            prompt = f"다음은 YouTube 동영상 전사본입니다:\n\n{complete_transcription}\n\n이 내용을 바탕으로 3개의 관련된 프로젝트 아이디어를 추천해 주세요."
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                project_ideas = response.choices[0].message.content
                st.write("추천 프로젝트 아이디어:")
                st.text_area("프로젝트 아이디어", project_ideas, height=150)
            except Exception as e:
                st.error(f"OpenAI API 오류: {e}")

            # 대체 주제 생성 기능 추가
            if st.button("새로운 주제 추천 받기"):
                prompt_alternative = f"다음은 YouTube 동영상 전사본입니다:\n\n{complete_transcription}\n\n이 내용을 바탕으로 3개의 새로운 관련 주제를 추천해 주세요."
                try:
                    response_alternative = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt_alternative}],
                        temperature=0.7,
                    )
                    alternative_topics = response_alternative.choices[0].message.content
                    st.write("추천 대체 주제:")
                    st.text_area("대체 주제", alternative_topics, height=150)
                except Exception as e:
                    st.error(f"OpenAI API 오류: {e}")