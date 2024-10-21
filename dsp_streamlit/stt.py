import os
import json
import re
import requests
import yt_dlp
import streamlit as st
from google.cloud import storage, speech
from openai import OpenAI
from dotenv import load_dotenv
import time

# .env 파일에서 환경 변수 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# 환경 변수에서 필요한 정보 가져오기
gcs_json_path = os.getenv("GCS_JSON_PATH")
speech_json_path = os.getenv("SPEECH_JSON_PATH")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
bucket_name = os.getenv("BUCKET_NAME")

# 필수 환경 변수 확인
if not gcs_json_path:
    raise ValueError("GCS_JSON_PATH is not set in the .env file")
if not speech_json_path:
    raise ValueError("SPEECH_JSON_PATH is not set in the .env file")
if not bucket_name:
    raise ValueError("BUCKET_NAME is not set in the .env file")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=openai_api_key)

# Google Cloud Storage 클라이언트 초기화
if not os.path.exists(gcs_json_path):
    raise FileNotFoundError(f"GCS JSON 파일이 존재하지 않습니다: {gcs_json_path}")

storage_client = storage.Client.from_service_account_json(gcs_json_path)

def get_top_videos(query, max_results=10, page_token=None):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&maxResults={max_results}&type=video&key={youtube_api_key}"
    if page_token:
        url += f"&pageToken={page_token}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    videos = data.get("items", [])
    next_page_token = data.get("nextPageToken")
    return [
        {
            "title": video["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={video['id']['videoId']}"
        }
        for video in videos if "videoId" in video["id"]
    ], next_page_token

def get_video_duration(video_url):
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict.get("duration", 0)

def extract_audio(video_url, title):
    sanitized_title = re.sub(r'[<>:"/\\|?*#]', '', title).strip()
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
        st.error(f"Error downloading audio: {e}")
        return None

def upload_to_gcs(file_path):
    try:
        blob = storage_client.bucket(bucket_name).blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        gcs_uri = f"gs://{bucket_name}/{os.path.basename(file_path)}"
        st.success(f"Uploaded to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        st.error(f"Error uploading to GCS: {e}")
        return None

def transcribe_audio(gcs_uri):
    client = speech.SpeechClient.from_service_account_json(speech_json_path)
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    
    operation = client.long_running_recognize(config=config, audio=audio)
    st.write("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    
    transcripts = [result.alternatives[0].transcript for result in response.results]
    return "\n".join(transcripts)

def brainstorm_project_ideas(transcriptions, existing_ideas=None):
    prompt = f"다음은 YouTube 동영상 전사본입니다:\n\n{transcriptions}\n"
    if existing_ideas:
        prompt += f"\n이전에 추천된 프로젝트 아이디어는 다음과 같습니다:\n{existing_ideas}\n"
    prompt += "\n이 내용을 바탕으로 몇 명의 AI 개발자가 서로 피드백을 주고받으며 해당 기술을 이용한 응용 프로그램 개발 프로젝트 3개를 브레인스토밍합니다. 각 개발자의 피드백을 반영하여 최종 프로젝트 아이디어를 추천해 주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API 오류: {e}")
        return "AI 프로젝트 아이디어 생성에 실패했습니다."

def main():
    st.title("YouTube Video to Audio Transcription")
    
    # Step 1: YouTube 검색어 입력
    query = st.text_input("검색어를 입력하세요:")
    
    # Step 2: YouTube 비디오 검색
    if query:
        max_videos = st.slider("추출할 비디오 개수를 선택하세요 (최대 6개):", 1, 6, 3)
        max_duration = st.slider("최대 비디오 길이 (초):", 30, 600, 300)
        if st.button("비디오 검색 및 오디오 추출"):
            all_transcriptions = ""
            progress_bar = st.progress(0)
            num_extracted = 0
            next_page_token = None

            while num_extracted < max_videos:
                videos, next_page_token = get_top_videos(query, max_results=10, page_token=next_page_token)
                if not videos:
                    st.warning("더 이상 검색 결과가 없습니다.")
                    break
                
                for video in videos:
                    if num_extracted >= max_videos:
                        break
                    
                    st.write(f"{video['title']}: {video['url']}")
                    video_duration = get_video_duration(video['url'])
                    if video_duration > max_duration:
                        st.warning(f"Skipping {video['title']} (Duration: {video_duration} seconds exceeds the limit of {max_duration} seconds)")
                        continue
                    audio_file = extract_audio(video['url'], video['title'])
                    if audio_file:
                        gcs_uri = upload_to_gcs(audio_file)
                        if gcs_uri:
                            transcriptions = transcribe_audio(gcs_uri)
                            st.text_area(f"Transcription for {video['title']}", transcriptions, height=200)
                            all_transcriptions += transcriptions + "\n"
                            num_extracted += 1
                    
                    # Update progress bar
                    progress = num_extracted / max_videos
                    progress_bar.progress(progress)

                if not next_page_token:
                    break

            if all_transcriptions:
                st.write("\n## 추천하는 프로젝트 아이디어:")
                project_ideas = brainstorm_project_ideas(all_transcriptions)
                st.session_state['project_ideas'] = project_ideas
                st.text_area("추천된 프로젝트 아이디어", project_ideas, height=300)

                # Step 3: 사용자 피드백 및 재생성 기능 추가
                if st.button("아이디어 재생성 요청"):
                    st.write("AI 개발자가 아이디어를 다시 브레인스토밍 중입니다...")
                    updated_project_ideas = brainstorm_project_ideas(all_transcriptions, existing_ideas=st.session_state['project_ideas'])
                    if 'project_ideas' in st.session_state:
                        st.session_state['project_ideas'] += "\n\n재생성된 아이디어:\n" + updated_project_ideas
                    else:
                        st.session_state['project_ideas'] = updated_project_ideas
                    st.text_area("업데이트된 프로젝트 아이디어", st.session_state['project_ideas'], height=300)

if __name__ == "__main__":
    main()
