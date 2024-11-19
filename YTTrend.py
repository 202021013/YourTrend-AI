import yt_dlp
import json
import numpy as np
import cv2
import openai
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
import re
import base64
from PIL import Image
import tempfile
from datetime import datetime
import os
import random
import time

class YouTubeShortsDownloader:
    def __init__(self):
        """Initialize the downloader with yt-dlp options"""
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }

    def get_trending_shorts(self, max_results=10):
        """Fetch trending YouTube Shorts using yt-dlp"""
        try:
            shorts_urls = []
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                playlist_urls = [
                    'https://www.youtube.com/hashtag/shorts/trending',
                    'https://www.youtube.com/shorts',
                ]
                
                for playlist_url in playlist_urls:
                    try:
                        print(f"Trying to fetch shorts from: {playlist_url}")
                        result = ydl.extract_info(playlist_url, download=False)
                        
                        if 'entries' in result:
                            for entry in result['entries']:
                                if entry and 'duration' in entry and entry['duration'] <= 60:
                                    video_url = f"https://youtube.com/shorts/{entry['id']}"
                                    if video_url not in shorts_urls:
                                        shorts_urls.append(video_url)
                                        
                                if len(shorts_urls) >= max_results:
                                    break
                            
                            if len(shorts_urls) >= max_results:
                                break
                    except Exception as e:
                        print(f"Error fetching from {playlist_url}: {e}")
                        continue
            
            return shorts_urls[:max_results]
            
        except Exception as e:
            print(f"Error in get_trending_shorts: {e}")
            return []

    def download_video(self, url, output_path="temp_videos"):
        """Download a single video using yt-dlp"""
        try:
            os.makedirs(output_path, exist_ok=True)
            video_id = url.split('/')[-1]
            output_template = os.path.join(output_path, f"{video_id}.mp4")
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            return output_template if os.path.exists(output_template) else None
            
        except Exception as e:
            print(f"Error downloading video {url}: {e}")
            return None

    def get_video_info(self, url):
        """Get video metadata using yt-dlp"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'upload_date': info.get('upload_date'),
                    'duration': info.get('duration'),
                }
        except Exception as e:
            print(f"Error getting video info for {url}: {e}")
            return None

class YouTubeShortsAnalyzer:
    def __init__(self, api_key):
        """Initialize the analyzer with OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
        self.embeddings_cache = {}
        self.downloader = YouTubeShortsDownloader()
        
    def extract_frames(self, video_path, num_frames=5):
        """Extract frames from video at regular intervals"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = total_frames // num_frames
            
            frames = []
            frame_positions = []
            
            for i in range(num_frames):
                frame_pos = i * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_positions.append(frame_pos)
            
            cap.release()
            return frames, frame_positions
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return None, None

    def encode_image(self, image_array):
        """Convert numpy array to base64 encoded image"""
        try:
            image = Image.fromarray(image_array)
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file, format='JPEG')
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            os.unlink(tmp_file_path)
            return encoded_string
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def vision_to_text(self, image_array):
        """Convert image to text description using OpenAI Vision API"""
        try:
            base64_image = self.encode_image(image_array)
            
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what's happening in this image from the YouTube Shorts video. Focus on the main subject, actions, and any visible trends or popular elements."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in vision to text conversion: {e}")
            return None

    def get_embedding(self, text):
        """Get embedding for text using OpenAI API"""
        try:
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response.data[0].embedding
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def analyze_content(self, text):
        """Analyze content using OpenAI API"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Extract key themes, topics, trends, and popular elements from this YouTube Shorts content description."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return None

    def visualize_trends(self, videos_data):
        """Visualize trends using t-SNE"""
        embeddings = [data['embedding'] for data in videos_data if 'embedding' in data]
        if not embeddings:
            return None
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        
        for i, data in enumerate(videos_data):
            plt.annotate(f"Video {i+1}", (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.title("YouTube Shorts Content Similarity Map")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"trends_visualization_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def extract_common_themes(self, analyses):
        """Extract common themes from content analyses"""
        all_text = " ".join(analyses)
        words = re.findall(r'\w+', all_text.lower())
        word_freq = Counter(words)
        
        stop_words = set(['the', 'is', 'at', 'which', 'and', 'or', 'in', 'to', 'of', 'a', 'an', 'this', 'that'])
        trending_topics = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 3}
        
        return dict(sorted(trending_topics.items(), key=lambda x: x[1], reverse=True)[:10])

    def process_single_video(self, url, video_path, video_info):
        """Process a single video and extract all relevant information"""
        try:
            # Extract frames
            frames, frame_positions = self.extract_frames(video_path)
            if not frames:
                return None
            
            # Process each frame
            video_descriptions = []
            for frame, position in zip(frames, frame_positions):
                description = self.vision_to_text(frame)
                if description:
                    video_descriptions.append({
                        'frame_position': position,
                        'description': description
                    })
            
            # Combine all descriptions
            combined_text = " ".join([d['description'] for d in video_descriptions])
            
            # Get embedding and analysis
            embedding = self.get_embedding(combined_text)
            analysis = self.analyze_content(combined_text)
            
            return {
                'url': url,
                'info': video_info,
                'frame_descriptions': video_descriptions,
                'embedding': embedding,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def analyze_shorts_batch(self, max_results=10):
        """Analyze a batch of YouTube Shorts"""
        print("Fetching trending shorts...")
        shorts_urls = self.downloader.get_trending_shorts(max_results)
        if not shorts_urls:
            print("No shorts found!")
            return None
            
        print(f"Found {len(shorts_urls)} shorts to analyze")
        
        videos_data = []
        for url in shorts_urls:
            try:
                print(f"\nProcessing: {url}")
                
                # Get video info
                video_info = self.downloader.get_video_info(url)
                if not video_info:
                    continue
                    
                # Download video
                video_path = self.downloader.download_video(url)
                if not video_path:
                    continue
                
                # Process video
                video_data = self.process_single_video(url, video_path, video_info)
                if video_data:
                    videos_data.append(video_data)
                
                # Cleanup video file
                if os.path.exists(video_path):
                    os.remove(video_path)
                
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
        
        if not videos_data:
            return None
            
        # Visualize trends
        visualization_path = self.visualize_trends(videos_data)
        
        # Extract common themes
        analyses = [data['analysis'] for data in videos_data if 'analysis' in data]
        trending_themes = self.extract_common_themes(analyses)
        
        results = {
            'videos_data': videos_data,
            'trending_themes': trending_themes,
            'visualization_path': visualization_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return results

def main():
    # OpenAI API 키 설정
    openai_api_key = "your-openai-api-key"
    
    # 분석기 초기화
    analyzer = YouTubeShortsAnalyzer(openai_api_key)
    
    try:
        # 쇼츠 분석
        print("Starting YouTube Shorts analysis...")
        results = analyzer.analyze_shorts_batch(max_results=10)
        
        if results:
            # 결과 출력
            print("\nTrending Themes:")
            for theme, count in results['trending_themes'].items():
                print(f"{theme}: {count} mentions")
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"shorts_analysis_results_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nVisualization saved as: {results['visualization_path']}")
            print(f"Full results saved to: {output_file}")
        else:
            print("Analysis failed or no results were obtained.")
            
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()