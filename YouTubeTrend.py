import os
import yt_dlp
import torch
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
import re

class YouTubeTrendAnalyzer:
    def __init__(self):
        # 다운로드 디렉토리 생성
        self.download_dir = 'downloads'
        os.makedirs(self.download_dir, exist_ok=True)
        
        # yt-dlp 기본 설정
        self.ydl_opts = {
            'format': 'best',  # 최고 품질의 단일 포맷 선택
            'outtmpl': os.path.join(self.download_dir, '%(title)s.%(ext)s'),
            'verbose': True,  # 자세한 로그 출력
        }
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 비전 모델 초기화
        print("���전 모델 로딩 중...")
        self.vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.vision_model.to(self.device)
        
        # 텍스트 임베딩 모델 초기화
        print("텍스트 임베딩 모델 로딩 중...")
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_embedder.to(self.device)
        
        print("초기화 완료.")

    def sanitize_filename(self, filename):
        """Create a safe filename from the video title"""
        # Replace non-ASCII characters with their closest ASCII equivalents
        filename = filename.encode('ascii', 'ignore').decode()
        # Remove or replace invalid filename characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limit length and remove trailing spaces
        filename = filename.strip()[:100]
        return filename

    def process_video(self, video_info):
        """비디오 처리 메인 파이프라인"""
        try:
            # Create safe filenames
            safe_title = self.sanitize_filename(video_info['title'])
            base_path = os.path.join(self.download_dir, safe_title)
            video_path = f"{base_path}.mp4"
            audio_path = f"{base_path}.wav"
            
            # Set up download options
            download_opts = self.ydl_opts.copy()
            download_opts['outtmpl'] = video_path
            
            # Download video and extract audio
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                print(f"Downloading video: {video_info['title']}")
                ydl.download([video_info['url']])
                
                # Verify files exist
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                print(f"Files downloaded successfully:\nVideo: {video_path}\nAudio: {audio_path}")
            
            try:
                # Process video frames
                print("Processing video frames...")
                vision_text = self.process_frames(video_path)
                print("Vision processing completed")
                
                # Process audio
                print("Processing audio...")
                speech_text = self.process_audio(audio_path)
                print("Speech processing completed")
                
                # Generate embeddings
                print("Generating embeddings...")
                with torch.amp.autocast(device_type=self.device.type):
                    vision_embedding = self.text_embedder.encode(vision_text, convert_to_tensor=True)
                    speech_embedding = self.text_embedder.encode(speech_text, convert_to_tensor=True)
                
                vision_embedding = vision_embedding.cpu().numpy()
                speech_embedding = speech_embedding.cpu().numpy()
                
                return {
                    'title': video_info['title'],
                    'vision_text': vision_text,
                    'speech_text': speech_text,
                    'vision_embedding': vision_embedding,
                    'speech_embedding': speech_embedding
                }
                
            finally:
                # Cleanup
                print("Cleaning up downloaded files...")
                for file_path in [video_path, audio_path]:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Removed: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error processing video {video_info['title']}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def visualize_embeddings(self, results):
        """임베딩 시각화"""
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("No valid results to visualize")
            return None
            
        try:
            vision_embeddings = np.array([r['vision_embedding'] for r in valid_results])
            speech_embeddings = np.array([r['speech_embedding'] for r in valid_results])
            
            pca = PCA(n_components=2)
            vision_2d = pca.fit_transform(vision_embeddings)
            speech_2d = pca.fit_transform(speech_embeddings)
            
            df_vision = pd.DataFrame(vision_2d, columns=['PC1', 'PC2'])
            df_vision['type'] = 'Vision'
            df_vision['title'] = [r['title'] for r in valid_results]
            
            df_speech = pd.DataFrame(speech_2d, columns=['PC1', 'PC2'])
            df_speech['type'] = 'Speech'
            df_speech['title'] = [r['title'] for r in valid_results]
            
            df = pd.concat([df_vision, df_speech])
            
            fig = px.scatter(df, x='PC1', y='PC2', 
                           color='type', hover_data=['title'],
                           title='YouTube Trending Videos - Vision vs Speech Embeddings')
            
            return fig
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return None

    def get_trending_videos(self, max_videos=2, region='KR'):
        """YouTube 트렌딩 비디오 정보 가져오기"""
        try:
            # 트렌딩 페이지용 옵션
            trending_opts = {
                'extract_flat': True,
                'quiet': False,  # 로그 출력 활성화
            }
            
            with yt_dlp.YoutubeDL(trending_opts) as ydl:
                print(f"{region} 지역의 트렌딩 비디오 검색 중...")
                
                # 트렌딩 페이지 URL
                trending_url = f"https://www.youtube.com/feed/trending?gl={region}"
                
                # 비디오 정보 추출
                result = ydl.extract_info(trending_url, download=False)
                
                videos = []
                if result and 'entries' in result:
                    for entry in result['entries'][:max_videos]:
                        if entry and 'id' in entry:
                            video_info = {
                                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                'title': entry.get('title', 'Untitled'),
                                'id': entry['id'],
                            }
                            print(f"발견된 비디오: {video_info['title']}")
                            videos.append(video_info)
                
                print(f"총 {len(videos)}개의 트렌딩 비디오를 찾았습니다.")
                return videos
                
        except Exception as e:
            print(f"트렌딩 비디오 검색 중 오류 발생: {str(e)}")
            return []

    def download_single_video(self, video_info):
        """단일 비디오 다운로드"""
        try:
            # 파일명에서 특수문자 제거
            safe_title = re.sub(r'[^\w\s-]', '', video_info['title'])
            video_path = os.path.join(self.download_dir, f"{safe_title}.mp4")
            
            # 다운로드 옵션 설정
            download_opts = {
                'format': 'best',  # 최고 품질의 단일 포맷
                'outtmpl': video_path,
                'verbose': True,
            }
            
            print(f"다운로드 시작: {video_info['title']}")
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(video_info['url'], download=True)
                print(f"다운로드 정보: {info.keys()}")
            
            if os.path.exists(video_path):
                print(f"다운로드 완료: {video_path}")
                return video_path
            else:
                print(f"파일을 찾을 수 없음: {video_path}")
                return None
                
        except Exception as e:
            print(f"비디오 다운로드 중 오류 발생: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return None

    def extract_frames(self, video_path, num_frames=5):
        """비디오에서 프레임 추출"""
        frames = []
        try:
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"전체 프레임 수: {total_frames}")
            
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            for idx in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                    print(f"프레임 추출: {idx}")
            
            video.release()
            return frames
            
        except Exception as e:
            print(f"프레임 추출 중 오류 발생: {str(e)}")
            if 'video' in locals():
                video.release()
            return frames

    def process_frames(self, video_path):
        """프레임 추출 및 캡셔닝"""
        try:
            print("프레임 추출 시작...")
            frames = self.extract_frames(video_path)
            
            if not frames:
                print("추출된 프레임이 없습니다.")
                return ""
            
            print("프레임 캡셔닝 시작...")
            captions = []
            
            with torch.amp.autocast(device_type=self.device.type):
                for i, image in enumerate(frames):
                    inputs = self.image_processor(image, return_tensors="pt").to(self.device)
                    output = self.vision_model.generate(**inputs, max_length=50)
                    caption = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    captions.append(caption)
                    print(f"프레임 {i+1} 캡션: {caption}")
            
            return " ".join(captions)
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
            return ""

    def process_audio(self, audio_path):
        """Process audio to extract text"""
        # This method should be implemented to process audio
        # For now, it returns a mock text
        return "Mock speech text"

def main():
    print("YouTube 트렌드 분석기 시작")
    analyzer = YouTubeTrendAnalyzer()
    
    # 트렌딩 비디오 가져오기 (테스트를 위해 2개만)
    trending_videos = analyzer.get_trending_videos(max_videos=2)
    
    if not trending_videos:
        print("트렌딩 비디오를 찾을 수 없습니다.")
        return
    
    for video in trending_videos:
        print(f"\n비디오 처리 시작: {video['title']}")
        
        # 비디오 다운로드
        video_path = analyzer.download_single_video(video)
        if not video_path:
            print("비디오 다운로드 실패, 다음 비디오로 넘어갑니다.")
            continue
        
        try:
            # 프레임 처리
            vision_text = analyzer.process_frames(video_path)
            print(f"비디오 분석 결과:\n{vision_text}")
            
        finally:
            # 임시 파일 정리
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"임시 파일 삭제 완료: {video_path}")
            except Exception as e:
                print(f"파일 정리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()