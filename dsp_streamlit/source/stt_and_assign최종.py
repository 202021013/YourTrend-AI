# 필요한 모듈 로드
import os
import json
import re
import requests
import yt_dlp
from google.cloud import storage, speech
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# 환경 변수에서 필요한 정보 가져오기
gcs_json_path = os.getenv("GCS_JSON_PATH")
speech_json_path = os.getenv("SPEECH_JSON_PATH")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
bucket_name = os.getenv("BUCKET_NAME")

# API 키가 없으면 사용자에게 입력 요청
if not youtube_api_key:
    youtube_api_key = input("YouTube API 키를 입력하세요: ")
if not openai_api_key:
    openai_api_key = input("OpenAI API 키를 입력하세요: ")

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
    """
    YouTube API를 사용하여 주어진 쿼리에 대한 상위 비디오를 검색합니다.
    
    :param query: 검색 쿼리
    :param max_results: 반환할 최대 비디오 수
    :param page_token: 다음 페이지 결과를 위한 토큰
    :return: 비디오 정보 리스트 (제목과 URL)와 다음 페이지 토큰
    """
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
    """
    YouTube 비디오의 길이를 초 단위로 가져옵니다.
    
    :param video_url: YouTube 비디오 URL
    :return: 비디오 길이 (초)
    """
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict.get("duration", 0)

def sanitize_title(title):
    """
    파일 이름으로 사용할 수 있도록 제목을 정리합니다.
    
    :param title: 원본 제목
    :return: 정리된 제목
    """
    title = re.sub(r'[<>:"/\\|?*#]', '', title)  # 특수문자 제거
    return title.strip()

def extract_audio(video_url, title):
    """
    YouTube 비디오에서 오디오를 추출하여 MP3 파일로 저장합니다.
    
    :param video_url: YouTube 비디오 URL
    :param title: 비디오 제목
    :return: 저장된 MP3 파일 경로 또는 None (실패 시)
    """
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
        print(f"Downloaded audio to: {output_path}.mp3")
        return f"{output_path}.mp3"
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def upload_to_gcs(file_path):
    """
    로컬 파일을 Google Cloud Storage에 업로드합니다.
    
    :param file_path: 업로드할 로컬 파일 경로
    :return: GCS URI 또는 None (실패 시)
    """
    try:
        blob = storage_client.bucket(bucket_name).blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        gcs_uri = f"gs://{bucket_name}/{os.path.basename(file_path)}"
        print(f"Uploaded to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def transcribe_audio(gcs_uri):
    """
    Google Speech-to-Text API를 사용하여 오디오를 텍스트로 변환합니다.
    
    :param gcs_uri: Google Cloud Storage에 있는 오디오 파일의 URI
    :return: 전사된 텍스트
    """
    client = speech.SpeechClient.from_service_account_json(speech_json_path)

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return "\n".join(transcripts)

def read_transcriptions(directory):
    """
    지정된 디렉토리에서 모든 전사 파일을 읽어 하나의 문자열로 결합합니다.
    
    :param directory: 전사 파일이 있는 디렉토리 경로
    :return: 모든 전사 내용을 포함한 문자열
    """
    transcriptions = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                transcriptions.append(file.read())
    return "\n".join(transcriptions)

def generate_project_ideas(transcriptions):
    """
    전사 내용을 바탕으로 GPT-4o-mini를 사용하여 프로젝트 아이디어를 생성합니다.
    
    :param transcriptions: 전사 내용
    :return: 생성된 프로젝트 아이디어
    """
    prompt = f"다음은 YouTube 동영상 전사본입니다:\n\n{transcriptions}\n\n이 내용을 바탕으로 3개의 관련된 프로젝트 아이디어를 추천해 주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return "AI 프로젝트 아이디어 생성에 실패했습니다."

def generate_project_output(project_idea, transcriptions):
    """
    선택된 프로젝트 아이디어에 대한 상세 계획과 예상 결과물을 생성합니다.
    
    :param project_idea: 선택된 프로젝트 아이디어
    :param transcriptions: 전사 내용
    :return: 생성된 프로젝트 계획 및 결과물
    """
    prompt = f"다음은 YouTube 동영상 전사본입니다:\n\n{transcriptions}\n\n선택된 프로젝트 아이디어: '{project_idea}'\n\n이 프로젝트 아이디어에 대한 구체적인 계획과 예상 결과물을 생성해 주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return "프로젝트 결과물 생성에 실패했습니다."
    
def generate_alternative_topics(transcriptions):
    """
    전사 내용을 바탕으로 GPT-4o-mini를 사용하여 대체 주제를 생성합니다.
    
    :param transcriptions: 전사 내용
    :return: 생성된 대체 주제 리스트
    """
    prompt = f"다음은 YouTube 동영상 전사본입니다:\n\n{transcriptions}\n\n이 내용을 바탕으로 3개의 새로운 관련 주제를 추천해 주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        topics = response.choices[0].message.content.split('\n')
        return [topic.strip() for topic in topics if topic.strip()]
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return ["대체 주제 생성에 실패했습니다."]

def main():
    """
    메인 실행 함수: 사용자 입력을 받아 전체 프로세스를 실행합니다.
    """
    query = input("검색어를 입력하세요: ")
    num_videos_to_extract = int(input("추출할 비디오 개수를 입력하세요 (최대 6개): "))
    num_videos_to_extract = min(num_videos_to_extract, 6)  # 최대 6개로 제한

    file_counter = 1
    found_videos = False
    next_page_token = None

    while file_counter <= num_videos_to_extract:
        videos, next_page_token = get_top_videos(query, page_token=next_page_token)

        # 비디오 처리 루프
        for video in videos:
            duration = get_video_duration(video['url'])
            if duration <= 120:
                found_videos = True
                print(f"Extracting audio from: {video['title']}")
                audio_file = extract_audio(video['url'], video['title'])
                if audio_file:
                    gcs_uri = upload_to_gcs(audio_file)
                    if gcs_uri:
                        try:
                            text = transcribe_audio(gcs_uri)
                            txt_file_path = os.path.join(os.getcwd(), "transcriptions", f"{file_counter}. {sanitize_title(video['title'])}.txt")
                            os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                            with open(txt_file_path, "w", encoding='utf-8') as f:
                                f.write(text)
                            print(f"Transcription saved to {txt_file_path}")

                            os.remove(audio_file)
                            print(f"Deleted audio file: {audio_file}")

                            file_counter += 1
                            if file_counter > num_videos_to_extract:
                                break
                        except Exception as e:
                            print(f"Error transcribing audio: {e}")
                    else:
                        print(f"Failed to upload audio for: {video['title']}")
                else:
                    print(f"Failed to extract audio for: {video['title']}")
            else:
                print(f"Skipping: {video['title']} (Duration: {duration} seconds)")

        if file_counter > num_videos_to_extract:
            break

        if not found_videos and next_page_token:
            user_choice = input("120초 이하의 비디오를 아직 찾지 못했습니다. 다음 페이지 결과를 검색하시겠습니까? (주의: 프로젝트와 연관성이 떨어질 수 있습니다) (예/아니오): ")
            if user_choice.lower() not in ['예', 'y', 'yes']:
                break
        elif not next_page_token:
            print("더 이상 검색 결과가 없습니다.")
            break

    if not found_videos:
        print("120초 이하의 비디오를 찾지 못했습니다. 다른 검색어를 시도해 주세요.")
        return

    # 전사본 읽기
    transcriptions_directory = os.path.join(os.getcwd(), "transcriptions")
    transcriptions = read_transcriptions(transcriptions_directory)

    while True:
        # 프로젝트 아이디어 생성
        print("\n추천하는 프로젝트 아이디어:")
        ideas = generate_project_ideas(transcriptions)
        print(ideas)

        # 사용자 선택
        user_choice = input("위 프로젝트 아이디어 중 하나를 선택하거나, 새로운 주제를 추천받으려면 '새로운 주제'라고 입력하세요: ")
        
        if user_choice.lower() == '새로운 주제':
            alternative_topics = generate_alternative_topics(transcriptions)
            print("\n새로운 주제 추천:")
            for i, topic in enumerate(alternative_topics, 1):
                print(f"{i}. {topic}")
            topic_choice = input("위 주제 중 하나를 선택하거나, 다시 프로젝트 아이디어를 보려면 '이전'이라고 입력하세요: ")
            if topic_choice.lower() == '이전':
                continue
            else:
                selected_idea = alternative_topics[int(topic_choice) - 1]
        else:
            selected_idea = user_choice

        # 프로젝트 결과물 생성
        project_output = generate_project_output(selected_idea, transcriptions)
        print(f"\n선택한 프로젝트 '{selected_idea}'의 결과물:\n{project_output}")

        # 사용자 만족도 확인
        satisfaction = input("이 프로젝트 결과물에 만족하십니까? (예/아니오): ")
        if satisfaction.lower() in ['예', 'y', 'yes']:
            print("좋습니다! 프로젝트를 즐겁게 진행하세요.")
            break
        else:
            print("죄송합니다. 다른 아이디어를 선택하거나 새로운 주제를 추천받아보세요.")

if __name__ == "__main__":
    main()