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
        st.write(f"Using device: {self.device}")
        
        with st.spinner('Loading models...'):
            self.vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vision_model.to(self.device)
            
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_embedder.to(self.device)

    def sanitize_filename(self, filename):
        filename = filename.encode('ascii', 'ignore').decode()
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        return filename.strip()[:100]

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
                                'title': entry.get('title', 'Untitled'),
                                'id': entry['id'],
                            }
                            st.write(f"Found video: {video_info['title']}")
                            videos.append(video_info)
                
                return videos
                
        except Exception as e:
            st.error(f"Error fetching trending videos: {str(e)}")
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
                    st.write(f"Extracted frame {idx}")
            
            video.release()
            return frames
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            if 'video' in locals():
                video.release()
            return frames

    def process_frames(self, video_path):
        try:
            frames = self.extract_frames(video_path)
            
            if not frames:
                st.warning("No frames extracted")
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
                    st.write(f"Frame {i+1} caption: {caption}")
            
            return " ".join(captions)
            
        except Exception as e:
            st.error(f"Error processing frames: {str(e)}")
            return ""

    def process_video(self, video_info):
        try:
            safe_title = self.sanitize_filename(video_info['title'])
            base_path = os.path.join(self.download_dir, safe_title)
            video_path = f"{base_path}.mp4"
            
            download_opts = self.ydl_opts.copy()
            download_opts['outtmpl'] = video_path
            
            with st.spinner('Downloading video...'):
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_info['url']])
            
            try:
                st.write("Processing video frames...")
                vision_text = self.process_frames(video_path)
                
                with torch.amp.autocast(device_type=self.device.type):
                    vision_embedding = self.text_embedder.encode(vision_text, convert_to_tensor=True)
                    speech_embedding = self.text_embedder.encode(vision_text, convert_to_tensor=True)  # Using vision text as placeholder
                
                vision_embedding = vision_embedding.cpu().numpy()
                speech_embedding = speech_embedding.cpu().numpy()
                
                return {
                    'title': video_info['title'],
                    'vision_text': vision_text,
                    'speech_text': vision_text,  # Placeholder
                    'vision_embedding': vision_embedding,
                    'speech_embedding': speech_embedding
                }
                
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.write(f"Traceback: {traceback.format_exc()}")
            return None

    def visualize_embeddings(self, results):
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            st.warning("No valid results to visualize")
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
                           title='YouTube Trending Videos Analysis')
            
            return fig
            
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            return None

def main():
    st.title("YouTube Trend Analyzer")
    
    st.sidebar.title("Settings")
    max_videos = st.sidebar.slider("Number of videos to analyze", 1, 10, 2)
    region = st.sidebar.selectbox("Region", ["KR", "US", "JP"], index=0)
    
    if st.sidebar.button("Start Analysis"):
        analyzer = YouTubeTrendAnalyzer()
        
        trending_videos = analyzer.get_trending_videos(max_videos=max_videos, region=region)
        
        if not trending_videos:
            st.error("No trending videos found")
            return
        
        results = []
        progress_bar = st.progress(0)
        
        for i, video in enumerate(trending_videos):
            st.subheader(f"Analyzing: {video['title']}")
            
            result = analyzer.process_video(video)
            if result:
                results.append(result)
                
                st.write("Vision Analysis:")
                st.write(result['vision_text'])
                
                st.write("Speech Analysis:")
                st.write(result['speech_text'])
            
            progress_bar.progress((i + 1) / len(trending_videos))
        
        if results:
            st.subheader("Embedding Visualization")
            fig = analyzer.visualize_embeddings(results)
            if fig:
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()