import streamlit as st
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class TrendAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
        }

    def get_trends(self, country='KR'):
        try:
            url = f"https://trends.google.co.kr/trends/trendingsearches/daily/rss?geo={country}"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'xml')
            
            trends = []
            scores = []
            items = soup.find_all('item')
            
            for item in items[:10]:
                title = item.find('title').text
                traffic = item.find('ht:approx_traffic').text
                traffic_num = int(traffic.replace(',', '').replace('+', ''))
                trends.append(title)
                scores.append(traffic_num)
                
            return trends, scores
        except Exception as e:
            st.error(f"RSS 피드 수집 오류 ({country}): {e}")
            return [], []

    def create_pie_chart(self, country):
        trends, scores = self.get_trends(country)
        
        if not trends:
            return None
            
        total = sum(scores)
        percentages = [(score/total*100) for score in scores]
        
        # 데이터 정렬
        sorted_indices = sorted(range(len(percentages)), 
                              key=lambda k: percentages[k],
                              reverse=True)
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_trends = [trends[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]

        # Plotly 파이 차트 생성
        fig = go.Figure(data=[go.Pie(
            labels=[f"#{i+1} {trend}" for i, trend in enumerate(sorted_trends)],
            values=sorted_scores,
            hovertemplate="<b>%{label}</b><br>" +
                         "검색량: %{value:,}<br>" +
                         "비율: %{percent:.1%}<br>" +
                         "<extra></extra>",
            textinfo='percent',
            hole=.3
        )])
        
        title = "한국" if country == "KR" else "미국"
        fig.update_layout(
            title=f"{title} 실시간 검색어 순위",
            showlegend=True,
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left" if country == "KR" else "right",
                x=-0.3 if country == "KR" else 1.5  # 1.3에서 1.5로 수정
            ),
            height=600
        )
        
        return fig
    
def main():
    st.set_page_config(page_title="구글 트렌드 모니터링", layout="wide")
    st.title("구글 트렌드 실시간 검색어 모니터링")
    
    analyzer = TrendAnalyzer()
    
    if st.button("새로고침"):
        st.rerun()  # experimental_rerun() 대신 rerun() 사용
    
    col1, col2 = st.columns(2)
    
    with col1:
        kr_fig = analyzer.create_pie_chart('KR')
        if kr_fig:
            st.plotly_chart(kr_fig, use_container_width=True)
            
    with col2:
        us_fig = analyzer.create_pie_chart('US')
        if us_fig:
            st.plotly_chart(us_fig, use_container_width=True)
    
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()