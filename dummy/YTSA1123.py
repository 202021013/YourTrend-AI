import cv2
import numpy as np
from openai import OpenAI
import json
from datetime import datetime
import time
from urllib.parse import urljoin
import random
import requests
import os
from bs4 import BeautifulSoup
import yt_dlp
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class YouTubeShortsVideoAnalyzer:
    def __init__(self, openai_api_key):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.setup_downloader()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        # downloads 디렉토리 생성
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
        
    def setup_downloader(self):
        """yt-dlp 설정"""
        self.ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }

    def get_trending_shorts(self, count=15):
        """인기 쇼츠 URL 수집"""
        shorts_data = []
        base_url = "https://www.youtube.com"
        trends_url = f"{base_url}/results?search_query=shorts&sp=CAMSAhAB"  # 조회수 기준 정렬
        
        try:
            response = requests.get(trends_url, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch shorts page: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 초기 데이터 스크립트 찾기
            for script in soup.find_all('script'):
                if 'ytInitialData' in str(script):
                    data_text = str(script).split('ytInitialData = ')[1].split(';</script>')[0]
                    try:
                        data = json.loads(data_text)
                        contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                        
                        for content in contents:
                            if 'itemSectionRenderer' in content:
                                items = content['itemSectionRenderer']['contents']
                                for item in items:
                                    if 'videoRenderer' in item:
                                        video = item['videoRenderer']
                                        video_id = video.get('videoId')
                                        title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                                        
                                        if video_id and '/shorts/' in video.get('navigationEndpoint', {}).get('commandMetadata', {}).get('webCommandMetadata', {}).get('url', ''):
                                            shorts_data.append({
                                                'video_id': video_id,
                                                'title': title,
                                                'video_url': f"{base_url}/shorts/{video_id}"
                                            })
                                            
                                            if len(shorts_data) >= count:
                                                break
                    except json.JSONDecodeError:
                        continue
                                            
            return shorts_data[:count]
            
        except Exception as e:
            print(f"Error fetching shorts: {str(e)}")
            return shorts_data

    def download_short(self, video_url):
        """쇼츠 다운로드"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                return os.path.join('downloads', f"{info['id']}.{info['ext']}")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None

    def extract_frames(self, video_path, num_frames=8):
        """비디오에서 프레임 추출"""
        if not os.path.exists(video_path):
            return None
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 영상 길이가 매우 짧을 경우 프레임 수 조정
        num_frames = min(num_frames, total_frames)
        frame_intervals = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_no in frame_intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                # 이미지 크기 조정 (메모리 효율성)
                frame = cv2.resize(frame, (480, 854))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(buffer.tobytes())
        
        cap.release()
        return frames

    def analyze_video(self, video_data):
        """비디오 분석"""
        try:
            print(f"Downloading and analyzing: {video_data['title']}")
            video_path = self.download_short(video_data['video_url'])
            if not video_path:
                return None
                
            frames = self.extract_frames(video_path)
            if not frames:
                if os.path.exists(video_path):
                    os.remove(video_path)
                return None
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",  # 수정된 부분: 최신 모델명으로 변경
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": """이 YouTube 쇼츠를 자세히 분석해주세요:
                                1. 영상 주제와 핵심 내용
                                2. 스토리텔링 방식과 전개
                                3. 사용된 편집기법과 특수효과
                                4. 인기 요인 분석:
                                   - 시청자 호기심 유발 요소
                                   - 공감대 형성 포인트
                                   - 바이럴 가능성
                                5. 타겟 시청자층 분석
                                6. 콘텐츠 카테고리와 장르
                                7. 크리에이터의 전략적 의도
                                8. 개선 가능한 부분이나 추천사항"""
                            }
                        ] + [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{frame}"
                            } for frame in frames
                        ]
                    }
                ],
                max_tokens=1500
            )
            
            # 임시 파일 삭제
            os.remove(video_path)
            
            return {
                'video_id': video_data['video_id'],
                'title': video_data['title'],
                'video_url': video_data['video_url'],
                'analysis': response.choices[0].message.content,
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            if os.path.exists(video_path):
                os.remove(video_path)
            return None

    def analyze_shorts_batch(self, shorts_data):
        """여러 쇼츠 분석"""
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
            
        analysis_results = []
        
        for short in shorts_data:
            result = self.analyze_video(short)
            if result:
                analysis_results.append(result)
                
            # API 레이트 리밋 방지
            time.sleep(random.uniform(2, 4))
            
        return analysis_results

    def save_results(self, analysis_results):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'shorts_video_analysis_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        return filename

    def cleanup(self):
        """리소스 정리"""
        if os.path.exists('downloads'):
            for file in os.listdir('downloads'):
                try:
                    os.remove(os.path.join('downloads', file))
                except Exception as e:
                    print(f"Error removing file {file}: {str(e)}")
            os.rmdir('downloads')

def main():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    analyzer = YouTubeShortsVideoAnalyzer(openai_api_key)
    
    try:
        print("인기 쇼츠 수집 중...")
        shorts_data = analyzer.get_trending_shorts(15)
        print(f"수집된 쇼츠 개수: {len(shorts_data)}")
        
        if not shorts_data:
            print("쇼츠를 찾을 수 없습니다.")
            return
            
        print("쇼츠 분석 중...")
        analysis_results = analyzer.analyze_shorts_batch(shorts_data)
        
        if analysis_results:
            filename = analyzer.save_results(analysis_results)
            print(f"분석 결과가 {filename}에 저장되었습니다.")
            print(f"총 {len(analysis_results)}개의 쇼츠가 분석되었습니다.")
        else:
            print("분석된 결과가 없습니다.")
            
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()