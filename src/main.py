import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path

from config import (
    DATA_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    BASE_DIR,
    RESULT_DIR
)
from data_loader import DocumentLoader
from search_engine import HybridSearchEngine
from analysis import InsightAnalyzer
from visualization import InsightVisualizer
from evaluation import InsightEvaluator

def setup_logging() -> None:
    """전역 로깅 설정"""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(handler)
        root_logger.setLevel(LOG_LEVEL)

def run_analysis_pipeline(query: str = "실제 데이터 분포를 반영한 사용자 리뷰 분석", save_results: bool = False) -> None:
    """
    전체 분석 파이프라인 실행

    Args:
        query (str, optional): 분석 쿼리. 
            Defaults to "실제 데이터 분포를 반영한 사용자 리뷰 분석"
        save_results (bool, optional): 분석 결과를 파일로 저장할지 여부.
            Defaults to False
    """
    logger = logging.getLogger(__name__)
    logger.info("분석 파이프라인 시작")

    try:
        # 결과 디렉토리 생성 (저장 옵션이 켜져 있는 경우에만)
        if save_results:
            os.makedirs(RESULT_DIR, exist_ok=True)
            logger.info(f"결과 디렉토리 확인: {RESULT_DIR}")

        # 1. 데이터 로딩
        loader = DocumentLoader(DATA_DIR)
        documents = loader.load_documents()
        logger.info(f"문서 로딩 완료: {len(documents)}개")

        # 2. 검색 엔진 초기화
        search_engine = HybridSearchEngine(documents)
        logger.info("검색 엔진 초기화 완료")

        # 3. 관련 문서 검색
        relevant_docs = search_engine.search(query)
        logger.info(f"관련 문서 검색 완료: {len(relevant_docs)}개")

        # 4. 통찰 분석
        analyzer = InsightAnalyzer()
        insights = analyzer.analyze(relevant_docs, query)
        logger.info(f"통찰 분석 완료: {len(insights)}개")

        # 5. 정량적 평가 수행
        evaluator = InsightEvaluator()
        evaluation_results = evaluator.evaluate(insights)
        logger.info(f"정량적 평가 완료: 종합 점수 {evaluation_results['overall_score']:.2f}/100")

        # 6. 결과 시각화 및 저장
        visualizer = InsightVisualizer()
        save_dir = RESULT_DIR if save_results else None
        visualizer.visualize_insights(insights, save_dir)
        
        # 7. 평가 결과 시각화 및 저장
        evaluator.visualize_evaluation(evaluation_results, save_dir)
        logger.info("시각화 완료")
        
        # 8. 분석 결과 저장 (JSON, CSV)
        if save_results:
            # JSON 저장
            json_path = save_analysis_results(insights, evaluation_results)
            # CSV 저장
            csv_path = evaluator.save_metrics_to_csv(evaluation_results, RESULT_DIR)
            logger.info(f"CSV 및 JSON 결과 파일 저장 완료: {csv_path}, {json_path}")

        logger.info("분석 파이프라인 성공적으로 완료")

    except FileNotFoundError as e:
        logger.error(f"데이터 디렉토리 또는 파일을 찾을 수 없습니다: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"분석 중 유효성 검사 실패: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}")
        raise

def save_analysis_results(insights: list, evaluation_results: dict) -> str:
    """
    분석 결과를 JSON 파일로 저장

    Args:
        insights (list): 생성된 인사이트 목록
        evaluation_results (dict): 평가 결과
        
    Returns:
        str: 저장된 JSON 파일 경로
    """
    logger = logging.getLogger(__name__)
    
    # 결과 디렉토리 생성 확인
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 저장할 결과 데이터 구성
    results = {
        "insights": insights,
        "evaluation": {
            "overall_score": float(evaluation_results["overall_score"]),
            "distribution_metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in evaluation_results["distribution_metrics"].items()
            },
            "category_coverage": evaluation_results["category_coverage"],
            "confidence_metrics": {
                "overall": evaluation_results["confidence_metrics"]["overall"],
                "consistency_score": float(evaluation_results["confidence_metrics"]["consistency_score"])
            }
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 결과 파일 저장
    result_path = os.path.join(RESULT_DIR, f"analysis_results_{timestamp}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"분석 결과가 JSON 파일로 저장되었습니다: {result_path}")
    
    return result_path


def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 데이터 디렉토리 확인
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")

        # 분석 파이프라인 실행
        run_analysis_pipeline(save_results=True)

    except Exception as e:
        logger.error(f"프로그램 실행 실패: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
