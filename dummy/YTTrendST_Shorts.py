import streamlit as st
import os
import yt_dlp
import torch
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
import re
import traceback

class YouTubeTrendAnalyzer:
    def __init__(self):
        self.download_dir = 'downloads'
        os.makedirs(self.download_dir, exist_ok=True)
        
        self.ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(self.download_dir, '%(title)s.%(ext)s'),
            'verbose': True,
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"사용 중인 디바이스: {self.device}")
        
        with st.spinner('모델 로딩 중...'):
            self.vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vision_model.to(self.device)
            
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_embedder.to(self.device)

    def sanitize_filename(self, filename):
        filename = filename.encode('ascii', 'ignore').decode()
        return re.sub(r'[^\w\s-]', '', filename).strip().lower()

    def get_trending_videos(self, max_videos=2, region='KR'):
        try:
            trending_opts = {
                'extract_flat': True,
                'quiet': False,
            }
            
            with yt_dlp.YoutubeDL(trending_opts) as ydl:
                trending_url = f"https://www.youtube.com/feed/trending?gl={region}"
                result = ydl.extract_info(trending_url, download=False)
                
                videos = []
                if result and 'entries' in result:
                    for entry in result['entries'][:max_videos]:
                        if entry and 'id' in entry:
                            video_info = {
                                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                'title': entry.get('title', '제목 없음'),
                                'id': entry['id'],
                            }
                            st.write(f"발견된 동영상: {video_info['title']}")
                            videos.append(video_info)
                
                return videos
                
        except Exception as e:
            st.error(f"트렌딩 동영상 가져오기 오류: {str(e)}")
            return []

    def extract_frames(self, video_path, num_frames=5):
        frames = []
        try:
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            for idx in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                    st.write(f"프레임 {idx} 추출 완료")
            
            video.release()
            return frames
            
        except Exception as e:
            st.error(f"프레임 추출 오류: {str(e)}")
            if 'video' in locals():
                video.release()
            return frames

    def process_frames(self, video_path):
        try:
            frames = self.extract_frames(video_path)
            
            if not frames:
                st.warning("추출된 프레임이 없습니다")
                return ""
            
            captions = []
            progress_bar = st.progress(0)
            
            with torch.amp.autocast(device_type=self.device.type):
                for i, image in enumerate(frames):
                    inputs = self.image_processor(image, return_tensors="pt").to(self.device)
                    output = self.vision_model.generate(**inputs, max_length=50)
                    caption = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    captions.append(caption)
                    progress_bar.progress((i + 1) / len(frames))
                    st.write(f"프레임 {i+1} 설명: {caption}")
            
            return " ".join(captions)
            
        except Exception as e:
            st.error(f"프레임 처리 오류: {str(e)}")
            return ""

    def process_video(self, video_info):
        try:
            safe_title = self.sanitize_filename(video_info['title'])
            base_path = os.path.join(self.download_dir, safe_title)
            video_path = f"{base_path}.mp4"
            
            download_opts = self.ydl_opts.copy()
            download_opts['outtmpl'] = video_path
            
            with st.spinner('동영상 다운로드 중...'):
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_info['url']])
            
            try:
                st.write("동영상 프레임 처리 중...")
                vision_text = self.process_frames(video_path)
                
                with torch.amp.autocast(device_type=self.device.type):
                    vision_embedding = self.text_embedder.encode(vision_text, convert_to_tensor=True)
                    speech_embedding = self.text_embedder.encode(vision_text, convert_to_tensor=True)
                
                vision_embedding = vision_embedding.cpu().numpy()
                speech_embedding = speech_embedding.cpu().numpy()
                
                return {
                    'title': video_info['title'],
                    'vision_text': vision_text,
                    'speech_text': vision_text,
                    'vision_embedding': vision_embedding,
                    'speech_embedding': speech_embedding,
                    'vision_analysis': vision_text,
                    'speech_analysis': vision_text
                }
                
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            st.error(f"동영상 처리 오류: {str(e)}")
            st.write(f"오류 추적: {traceback.format_exc()}")
            return None

    def visualize_embeddings(self, results):
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) < 2:
            st.warning("시각화를 위해서는 최소 2개의 유효한 동영상이 필요합니다. 더 많은 동영상을 분석해주세요.")
            return None
            
        try:
            vision_embeddings = np.array([r['vision_embedding'] for r in valid_results])
            speech_embeddings = np.array([r['speech_embedding'] for r in valid_results])
            
            n_components = min(2, len(valid_results) - 1)
            
            pca = PCA(n_components=n_components)
            vision_2d = pca.fit_transform(vision_embeddings)
            speech_2d = pca.fit_transform(speech_embeddings)
            
            df_vision = pd.DataFrame(vision_2d, columns=['PC1', 'PC2'] if n_components == 2 else ['PC1'])
            df_vision['type'] = '영상'
            df_vision['title'] = [r['title'] for r in valid_results]
            
            df_speech = pd.DataFrame(speech_2d, columns=['PC1', 'PC2'] if n_components == 2 else ['PC1'])
            df_speech['type'] = '음성'
            df_speech['title'] = [r['title'] for r in valid_results]
            
            df = pd.concat([df_vision, df_speech])
            
            if n_components == 1:
                df['PC2'] = 0
                
            fig = px.scatter(df, x='PC1', y='PC2', 
                            color='type', hover_data=['title'],
                            title='YouTube 트렌딩 동영상 분석')
            
            return fig
            
        except Exception as e:
            st.error(f"시각화 오류: {str(e)}")
            return None

def main():
    st.title("YouTube 트렌드 분석기")
    
    st.sidebar.title("설정")
    max_videos = st.sidebar.slider("분석할 동영상 수", 1, 10, 2)
    region = st.sidebar.selectbox("지역", ["KR", "US", "JP"], index=0)
    
    if st.sidebar.button("분석 시작"):
        analyzer = YouTubeTrendAnalyzer()
        
        trending_videos = analyzer.get_trending_videos(max_videos=max_videos, region=region)
        
        if not trending_videos:
            st.error("트렌딩 동영상을 찾을 수 없습니다")
            return
        
        results = []
        progress_bar = st.progress(0)
        
        for i, video in enumerate(trending_videos):
            st.subheader(f"분석 중: {video['title']}")
            
            analysis = analyzer.process_video(video)
            if analysis:
                results.append(analysis)
                
                st.write("영상 분석:")
                if 'vision_analysis' in analysis:
                    st.write(analysis['vision_analysis'])
                else:
                    st.write("영상 분석 데이터를 찾을 수 없습니다.")
                
                st.write("음성 분석:")
                if 'speech_analysis' in analysis:
                    st.write(analysis['speech_analysis'])
                else:
                    st.write("음성 분석 데이터를 찾을 수 없습니다.")
            
            progress_bar.progress((i + 1) / len(trending_videos))
        
        if results:
            st.subheader("임베딩 시각화")
            fig = analyzer.visualize_embeddings(results)
            if fig:
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()