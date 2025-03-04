"""
리뷰 분석 시스템의 설정을 관리하는 모듈
"""
from typing import Dict, Final
from pathlib import Path

# 카테고리 설정
CATEGORIES: Final[list] = ["불만 사항", "개선 요구", "기술적 문제", "장점"]
CATEGORY_COLORS: Final[Dict[str, str]] = {
    "불만 사항": "#66b3ff",
    "개선 요구": "#99ff99",
    "기술적 문제": "#ff9999",
    "장점": "#ffcc99"
}

# 실제 데이터 분포
REAL_DISTRIBUTION: Final[Dict[str, float]] = {
    "불만 사항": 53.0,
    "개선 요구": 27.0,
    "기술적 문제": 16.5,
    "장점": 3.5
}

# LLM 설정
MODEL_NAME: Final[str] = "gpt-4-turbo"
CACHE_TTL_HOURS: Final[int] = 1

# 검색 설정
BM25_TOP_K: Final[int] = 3
FAISS_TOP_K: Final[int] = 3
RETRIEVER_WEIGHTS: Final[Dict[str, float]] = {
    "bm25": 0.4,
    "faiss": 0.6
}

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "review_data"
RESULT_DIR = BASE_DIR.parent / "result"

# 로깅 설정
LOG_FORMAT: Final[str] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL: Final[str] = 'INFO'

# 시각화 설정
FIGURE_SIZE: Final[tuple] = (12, 6)
PLOT_ALPHA: Final[float] = 0.6
GRID_ALPHA: Final[float] = 0.3