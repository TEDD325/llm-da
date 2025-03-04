"""
정량적 평가 지표를 계산하는 모듈
생성된 인사이트 분포와 실제 분포 간의 차이를 측정하고 평가
"""
from typing import List, Dict, Any, Tuple
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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

class InsightEvaluator:
    """인사이트 생성 결과의 정량적 평가를 위한 클래스"""

    def __init__(self):
        """평가기 초기화"""
        self._setup_logging()
        self.logger.info("인사이트 평가기 초기화")
        
    def _setup_logging(self) -> None:
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(LOG_LEVEL)
    
    def evaluate(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        생성된 인사이트에 대한 종합적 평가 수행

        Args:
            insights (List[Dict[str, Any]]): 평가할 인사이트 리스트

        Returns:
            Dict[str, Any]: 다양한 평가 지표를 포함한 평가 결과
        """
        try:
            self.logger.info("인사이트 평가 시작")
            
            # 인사이트가 비어있는 경우 예외 처리
            if not insights:
                raise ValueError("평가할 인사이트가 없습니다")
            
            # 분포 계산
            distribution_metrics = self._evaluate_distribution(insights)
            
            # 신뢰도 점수 평가
            confidence_metrics = self._evaluate_confidence(insights)
            
            # 카테고리 포함 여부 평가
            category_coverage = self._evaluate_category_coverage(insights)
            
            # 종합 평가 지표
            overall_score = self._calculate_overall_score(
                distribution_metrics,
                confidence_metrics,
                category_coverage
            )
            
            # 결과 통합
            evaluation_results = {
                "distribution_metrics": distribution_metrics,
                "confidence_metrics": confidence_metrics,
                "category_coverage": category_coverage,
                "overall_score": overall_score
            }
            
            self.logger.info(f"인사이트 평가 완료: 종합 점수 {overall_score:.2f}/100")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"평가 중 오류 발생: {str(e)}")
            raise
    
    def _evaluate_distribution(self, insights: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        실제 분포와 생성된 분포 간의 차이를 평가

        Args:
            insights (List[Dict[str, Any]]): 평가할 인사이트 리스트

        Returns:
            Dict[str, float]: 분포 관련 평가 지표
        """
        # 생성된 분포 계산
        df = pd.DataFrame(insights)
        category_counts = df['insight_type'].value_counts()
        total_insights = len(df)
        
        generated_dist = {
            cat: (category_counts.get(cat, 0) / total_insights) * 100 
            for cat in CATEGORIES
        }
        
        # 실제 분포와 생성된 분포를 리스트로 변환
        real_values = [REAL_DISTRIBUTION[cat] for cat in CATEGORIES]
        generated_values = [generated_dist[cat] for cat in CATEGORIES]
        
        # 정량적 지표 계산
        rmse = np.sqrt(mean_squared_error(real_values, generated_values))
        mae = mean_absolute_error(real_values, generated_values)
        r2 = r2_score(real_values, generated_values)
        
        # 분포 유사도 지표 (Jensen-Shannon Divergence의 간소화된 버전)
        # 값이 작을수록 분포가 유사함
        real_norm = np.array(real_values) / sum(real_values)
        gen_norm = np.array(generated_values) / sum(generated_values)
        distribution_divergence = np.sum(np.abs(real_norm - gen_norm)) / 2
        
        # 분포 정확도 점수 (100점 만점)
        # 완벽하게 일치하면 100점, 완전히 다르면 0점
        distribution_accuracy = (1 - distribution_divergence) * 100
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "distribution_divergence": distribution_divergence,
            "distribution_accuracy": distribution_accuracy
        }
    
    def _evaluate_confidence(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        인사이트의 신뢰도 점수에 대한 통계적 평가

        Args:
            insights (List[Dict[str, Any]]): 평가할 인사이트 리스트

        Returns:
            Dict[str, Any]: 신뢰도 관련 평가 지표
        """
        df = pd.DataFrame(insights)
        confidence_scores = df['confidence_score'].values
        
        # 카테고리별 평균 신뢰도
        category_confidence = {}
        for cat in CATEGORIES:
            cat_scores = df[df['insight_type'] == cat]['confidence_score'].values
            if len(cat_scores) > 0:
                category_confidence[cat] = {
                    "mean": float(np.mean(cat_scores)),
                    "min": float(np.min(cat_scores)),
                    "max": float(np.max(cat_scores)),
                    "std": float(np.std(cat_scores))
                }
            else:
                category_confidence[cat] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0
                }
        
        # 전체 신뢰도 지표
        overall_confidence = {
            "mean": float(np.mean(confidence_scores)),
            "min": float(np.min(confidence_scores)),
            "max": float(np.max(confidence_scores)),
            "std": float(np.std(confidence_scores))
        }
        
        # 신뢰도 점수의 일관성 (표준편차 기반)
        # 표준편차가 낮을수록 일관된 신뢰도를 보임
        confidence_consistency = 100 * (1 - min(1.0, overall_confidence["std"] * 2))
        
        return {
            "overall": overall_confidence,
            "by_category": category_confidence,
            "consistency_score": confidence_consistency
        }
    
    def _evaluate_category_coverage(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        모든 카테고리가 적절하게 포함되었는지 평가

        Args:
            insights (List[Dict[str, Any]]): 평가할 인사이트 리스트

        Returns:
            Dict[str, Any]: 카테고리 포함 관련 평가 지표
        """
        df = pd.DataFrame(insights)
        categories_found = set(df['insight_type'].unique())
        
        # 각 카테고리 포함 여부
        categories_included = {cat: cat in categories_found for cat in CATEGORIES}
        
        # 카테고리 포함률 (%)
        coverage_percentage = (len(categories_found) / len(CATEGORIES)) * 100
        
        # 누락된 카테고리
        missing_categories = list(set(CATEGORIES) - categories_found)
        
        return {
            "categories_included": categories_included,
            "coverage_percentage": coverage_percentage,
            "missing_categories": missing_categories
        }
    
    def _calculate_overall_score(
        self, 
        distribution_metrics: Dict[str, float],
        confidence_metrics: Dict[str, Any],
        category_coverage: Dict[str, Any]
    ) -> float:
        """
        종합 평가 점수 계산 (100점 만점)

        Args:
            distribution_metrics (Dict[str, float]): 분포 평가 지표
            confidence_metrics (Dict[str, Any]): 신뢰도 평가 지표
            category_coverage (Dict[str, Any]): 카테고리 포함 평가 지표

        Returns:
            float: 종합 평가 점수 (0-100)
        """
        # 가중치 설정
        weights = {
            "distribution": 0.5,  # 분포 정확도 가중치
            "confidence": 0.3,    # 신뢰도 가중치
            "coverage": 0.2       # 카테고리 포함 가중치
        }
        
        # 각 부분 점수 계산
        distribution_score = distribution_metrics["distribution_accuracy"]
        confidence_score = confidence_metrics["consistency_score"]
        coverage_score = category_coverage["coverage_percentage"]
        
        # 종합 점수 계산
        overall_score = (
            weights["distribution"] * distribution_score +
            weights["confidence"] * confidence_score +
            weights["coverage"] * coverage_score
        )
        
        return overall_score
    
    def visualize_evaluation(self, evaluation_results: Dict[str, Any], save_dir: str = None) -> Tuple[Figure, List[Axes]]:
        """
        평가 결과를 시각화

        Args:
            evaluation_results (Dict[str, Any]): 평가 결과
            save_dir (str, optional): 결과를 저장할 디렉토리 경로. 기본값은 None.

        Returns:
            Tuple[Figure, List[Axes]]: 생성된 그림과 축 객체
        """
        # 그림 설정
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 분포 정확도 게이지 차트
        self._plot_gauge_chart(
            axs[0, 0], 
            evaluation_results["distribution_metrics"]["distribution_accuracy"],
            "분포 정확도 점수",
            0, 100
        )
        
        # 2. 종합 평가 점수 게이지 차트
        self._plot_gauge_chart(
            axs[0, 1], 
            evaluation_results["overall_score"],
            "종합 평가 점수",
            0, 100
        )
        
        # 3. 카테고리별 신뢰도 막대 그래프
        self._plot_category_confidence(
            axs[1, 0],
            evaluation_results["confidence_metrics"]["by_category"]
        )
        
        # 4. 카테고리 포함 여부 막대 그래프
        self._plot_category_coverage(
            axs[1, 1],
            evaluation_results["category_coverage"]["categories_included"]
        )
        
        plt.tight_layout()
        
        # 그래프 저장 (선택적)
        if save_dir:
            self._save_figure(fig, save_dir, "evaluation_metrics")
            
        return fig, axs
        
    def _save_figure(self, fig: Figure, save_dir: str, base_name: str) -> str:
        """
        그래프를 파일로 저장

        Args:
            fig (Figure): 저장할 그래프
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
        
        self.logger.info(f"평가 그래프가 저장되었습니다: {file_path}")
        return file_path
    
    def _plot_gauge_chart(self, ax: plt.Axes, value: float, title: str, 
                          min_val: float = 0, max_val: float = 100) -> None:
        """
        게이지 차트 그리기

        Args:
            ax (plt.Axes): 그래프를 그릴 축
            value (float): 표시할 값
            title (str): 차트 제목
            min_val (float): 최소값
            max_val (float): 최대값
        """
        # 게이지 차트 범위 및 각도 설정
        angles = np.linspace(0, 180, 100)
        
        # 값을 각도로 변환
        normalized_value = (value - min_val) / (max_val - min_val)
        angle = 180 * normalized_value
        
        # 배경 게이지 (회색)
        ax.plot(angles, [1] * len(angles), color='lightgray', linewidth=15)
        
        # 값 게이지 (파란색에서 초록색 그라데이션)
        value_angles = np.linspace(0, angle, int(angle) + 1)
        
        # 색상 설정: 점수에 따라 색상 변경
        if value < 50:
            color = '#FF5C5C'  # 빨간색 (낮은 점수)
        elif value < 75:
            color = '#FFA500'  # 주황색 (중간 점수)
        else:
            color = '#4CAF50'  # 초록색 (높은 점수)
            
        ax.plot(value_angles, [1] * len(value_angles), color=color, linewidth=15)
        
        # 텍스트 추가
        ax.text(90, 0.5, f'{value:.1f}', ha='center', va='center', fontsize=24)
        
        # 차트 정리
        ax.set_title(title, fontsize=16)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(-10, 190)
        ax.set_frame_on(False)
        ax.set_yticks([])
        ax.set_xticks([])
        
        # 눈금 표시
        for i in range(0, 181, 30):
            ax.plot([i, i], [0.9, 1.1], color='gray')
            ax.text(i, 0.8, str(int(min_val + (i/180) * (max_val - min_val))), 
                   ha='center', fontsize=10)
    
    def _plot_category_confidence(self, ax: plt.Axes, 
                                 category_confidence: Dict[str, Dict[str, float]]) -> None:
        """
        카테고리별 신뢰도 막대 그래프 생성

        Args:
            ax (plt.Axes): 그래프를 그릴 축
            category_confidence (Dict[str, Dict[str, float]]): 카테고리별 신뢰도 정보
        """
        # 데이터 추출
        categories = list(category_confidence.keys())
        means = [category_confidence[cat]["mean"] for cat in categories]
        stds = [category_confidence[cat]["std"] for cat in categories]
        
        # 막대 그래프
        x = range(len(categories))
        bars = ax.bar(x, means, yerr=stds, align='center', alpha=0.7,
                     color=[CATEGORY_COLORS[cat] for cat in categories],
                     capsize=5)
        
        # 그래프 스타일링
        ax.set_title('카테고리별 평균 신뢰도')
        ax.set_ylabel('신뢰도 점수')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=GRID_ALPHA)
        
        # 값 표시
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean:.2f}', ha='center', va='bottom')
    
    def _plot_category_coverage(self, ax: plt.Axes, 
                               categories_included: Dict[str, bool]) -> None:
        """
        카테고리 포함 여부 막대 그래프 생성

        Args:
            ax (plt.Axes): 그래프를 그릴 축
            categories_included (Dict[str, bool]): 카테고리 포함 여부
        """
        # 데이터 추출
        categories = list(categories_included.keys())
        included = [int(categories_included[cat]) for cat in categories]
        
        # 막대 그래프
        x = range(len(categories))
        bars = ax.bar(x, included, align='center', alpha=0.7,
                     color=[CATEGORY_COLORS[cat] if categories_included[cat] 
                            else 'lightgray' for cat in categories])
        
        # 그래프 스타일링
        ax.set_title('카테고리 포함 여부')
        ax.set_ylabel('포함 (1 = 예, 0 = 아니오)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=GRID_ALPHA)
        
        # 포함/미포함 라벨 표시
        for bar, is_included in zip(bars, included):
            label = "포함" if is_included else "미포함"
            color = "white" if is_included else "black"
            ax.text(bar.get_x() + bar.get_width()/2., 0.5,
                   label, ha='center', va='center', color=color)
                   
    def save_metrics_to_csv(self, evaluation_results: Dict[str, Any], result_dir: str) -> str:
        """
        평가 메트릭을 CSV 파일로 저장
        
        Args:
            evaluation_results (Dict[str, Any]): 평가 결과
            result_dir (str): 결과 저장 디렉토리
            
        Returns:
            str: 저장된 CSV 파일 경로
        """
        # 결과 디렉토리 생성
        os.makedirs(result_dir, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(result_dir, f"evaluation_metrics_{timestamp}.csv")
        
        # 평평한 구조로 메트릭 변환
        flat_metrics = {}
        
        # 전체 점수
        flat_metrics['overall_score'] = evaluation_results['overall_score']
        
        # 분포 메트릭
        for key, value in evaluation_results['distribution_metrics'].items():
            flat_metrics[f'distribution_{key}'] = value
        
        # 신뢰도 메트릭
        # 전체 신뢰도
        for key, value in evaluation_results['confidence_metrics']['overall'].items():
            flat_metrics[f'confidence_overall_{key}'] = value
        
        # 일관성 점수
        flat_metrics['confidence_consistency_score'] = evaluation_results['confidence_metrics']['consistency_score']
        
        # 카테고리별 신뢰도
        for category, metrics in evaluation_results['confidence_metrics']['by_category'].items():
            for key, value in metrics.items():
                flat_metrics[f'confidence_{category}_{key}'] = value
        
        # 카테고리 포함 지표
        flat_metrics['category_coverage_percentage'] = evaluation_results['category_coverage']['coverage_percentage']
        for category, included in evaluation_results['category_coverage']['categories_included'].items():
            flat_metrics[f'category_included_{category}'] = 1 if included else 0
        
        # CSV 파일 저장
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 행
            writer.writerow(['Metric', 'Value'])
            # 데이터 행
            for key, value in flat_metrics.items():
                writer.writerow([key, value])
        
        self.logger.info(f"평가 메트릭이 CSV 파일에 저장되었습니다: {csv_path}")
        return csv_path
