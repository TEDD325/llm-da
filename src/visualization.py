"""
분석 결과 시각화를 담당하는 모듈
카테고리 분포와 신뢰도 점수를 시각화
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import platform

from config import (
    CATEGORIES,
    CATEGORY_COLORS,
    REAL_DISTRIBUTION,
    FIGURE_SIZE,
    PLOT_ALPHA,
    GRID_ALPHA,
    LOG_FORMAT,
    LOG_LEVEL
)

class InsightVisualizer:
    """통찰 결과 시각화 클래스"""

    def __init__(self):
        """시각화 도구 초기화"""
        self._setup_logging()
        self.logger.info("시각화 도구 초기화")
        # 'seaborn'이 유효하지 않은 스타일로 나타나 'ggplot'으로 변경
        plt.style.use('ggplot')
        
        # 한글 폰트 설정
        self._setup_korean_font()

    def _setup_logging(self) -> None:
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False  # Propagation 비활성화
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(LOG_LEVEL)
            
    def _setup_korean_font(self) -> None:
        """한글 폰트 설정"""
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            self.logger.info("한글 폰트 설정: macOS")
            plt.rc('font', family='AppleGothic')
        elif system == 'Windows':
            self.logger.info("한글 폰트 설정: Windows")
            plt.rc('font', family='Malgun Gothic')
        elif system == 'Linux':
            self.logger.info("한글 폰트 설정: Linux")
            plt.rc('font', family='NanumGothic')
        
        # 음수 표시 문제 해결
        plt.rc('axes', unicode_minus=False)
        
        # 폰트 캐시 새로고침 - 안전한 방식 사용
        try:
            # matplotlib 버전에 따라 _rebuild()가 있을 수도 있고 없을 수도 있음
            mpl.font_manager._rebuild()
        except AttributeError:
            self.logger.info("폰트 캐시 재설정 대체 방법 사용")
            # 폰트 캐시 재설정 대체 방법
            mpl.font_manager.fontManager = mpl.font_manager.FontManager()

    def visualize_insights(self, insights: List[Dict[str, Any]], save_dir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        통찰 결과 시각화 수행

        Args:
            insights (List[Dict[str, Any]]): 시각화할 통찰 리스트
            save_dir (Optional[str]): 결과를 저장할 디렉토리 경로
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: 생성된 그림과 축 객체
        """
        try:
            self.logger.info("통찰 시각화 시작")
            df = pd.DataFrame(insights)
            
            # 단일 그래프만 생성 (박스 플롯 제거)
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            self._plot_category_distribution(df, ax)
            
            plt.tight_layout()
            self.logger.info("시각화 완료")
            
            # 결과 저장 (선택적)
            if save_dir:
                self.save_figure(fig, save_dir, "insight_distribution")
                
            plt.show()
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {str(e)}")
            raise

    def _plot_category_distribution(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        카테고리 분포 시각화

        Args:
            df (pd.DataFrame): 통찰 데이터프레임
            ax (plt.Axes): 그래프를 그릴 축
        """
        self.logger.debug("카테고리 분포 시각화")
        
        # 실제 분포와 생성된 분포 계산
        category_counts = df['insight_type'].value_counts()
        total_insights = len(df)
        
        generated_dist = {
            cat: (category_counts.get(cat, 0) / total_insights) * 100 
            for cat in CATEGORIES
        }
        
        # 막대 그래프 위치 설정
        x = range(len(CATEGORIES))
        width = 0.35
        
        # 실제 분포 막대 그래프 - 짙은 색상으로 표시
        ax.bar([i - width/2 for i in x], 
               [REAL_DISTRIBUTION[cat] for cat in CATEGORIES],
               width, label='실제 분포', alpha=0.9,
               color=[CATEGORY_COLORS[cat] for cat in CATEGORIES])
        
        # 생성된 분포 막대 그래프 - 연한 색상과 패턴으로 구분
        bars = ax.bar([i + width/2 for i in x],
               [generated_dist[cat] for cat in CATEGORIES],
               width, label='생성된 분포', alpha=0.65,
               color=[self._adjust_color_alpha(CATEGORY_COLORS[cat], 0.5) for cat in CATEGORIES],
               hatch='////')
        
        # 그래프 스타일링
        ax.set_title('카테고리별 분포 비교')
        ax.set_xlabel('카테고리')
        ax.set_ylabel('비율 (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, rotation=45)
        ax.legend()
        ax.grid(True, alpha=GRID_ALPHA)

    def _plot_confidence_scores(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        신뢰도 점수 분포 시각화

        Args:
            df (pd.DataFrame): 통찰 데이터프레임
            ax (plt.Axes): 그래프를 그릴 축
        """
        self.logger.debug("신뢰도 점수 시각화")
        
        # 카테고리별 신뢰도 점수 추출
        scores_by_category = {
            cat: df[df['insight_type'] == cat]['confidence_score'].tolist()
            for cat in CATEGORIES
        }
        
        # 박스플롯 생성
        box_colors = [CATEGORY_COLORS[cat] for cat in CATEGORIES]
        ax.boxplot([scores_by_category[cat] for cat in CATEGORIES],
                  labels=CATEGORIES,
                  patch_artist=True,
                  boxprops=dict(alpha=PLOT_ALPHA),
                  medianprops=dict(color="black"),
                  flierprops=dict(marker='o', markerfacecolor='gray'))
        
        # 그래프 스타일링
        ax.set_title('카테고리별 신뢰도 점수 분포')
        ax.set_xlabel('카테고리')
        ax.set_ylabel('신뢰도 점수')
        ax.set_xticklabels(CATEGORIES, rotation=45)
        ax.grid(True, alpha=GRID_ALPHA)
        
        # 각 카테고리의 박스 색상 설정
        for patch, color in zip(ax.patches, box_colors):
            patch.set_facecolor(color)

    @staticmethod
    def _adjust_color_alpha(color: str, alpha: float = 0.7) -> str:
        """
        색상의 투명도 조정

        Args:
            color (str): 원본 색상 (hex 형식)
            alpha (float): 적용할 투명도 (0~1)

        Returns:
            str: 투명도가 조정된 색상
        """
        return f"{color}{int(alpha*255):02x}"
        
    def save_figure(self, fig: plt.Figure, save_dir: str, base_name: str) -> str:
        """
        그래프를 파일로 저장

        Args:
            fig (plt.Figure): 저장할 그래프
            save_dir (str): 저장 디렉토리
            base_name (str): 파일 기본 이름

        Returns:
            str: 저장된 파일 경로
        """
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(save_dir, f"{base_name}_{timestamp}.png")
        
        # 그래프 저장 (PNG 형식)
        fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
        
        self.logger.info(f"그래프가 저장되었습니다: {file_path}")
        return file_path