# 모듈 2: 데이터 파이프라인 구축

## 학습 목표
- 다양한 데이터 소스와 포맷에서 문서를 효율적으로 로드하는 방법 이해
- 문서 처리를 위한 일관된 파이프라인 구축 방법 습득
- Airflow를 활용한 데이터 수집 및 전처리 자동화 기법 학습

## 1. 데이터 소스 다양성 관리

### 다중 포맷 처리 전략

**지원 가능한 문서 형식:**
- 일반 텍스트 파일 (.txt)
- 구조화된 데이터 (.json, .csv)
- 웹 페이지 콘텐츠 (HTML)
- 문서 파일 (.pdf, .docx)

### 통합 로더 아키텍처

```python
# 확장 가능한 문서 로더 인터페이스
class DocumentLoader:
    """문서 로딩을 처리하는 클래스"""
    
    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir (Path): 문서가 저장된 기본 디렉토리 경로
        """
        self.base_dir = base_dir
        self._setup_logging()

    def load_documents(self, directory=None):
        """
        지정된 디렉토리에서 문서를 로드

        Args:
            directory (Optional[str]): 로드할 디렉토리 경로. None이면 base_dir 사용

        Returns:
            List[Document]: 로드된 문서 리스트
        """
        target_dir = Path(directory) if directory else self.base_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target_dir}")
        
        documents = []
        # 확장자별 처리
        for file_path in target_dir.glob("*.txt"):
            try:
                documents.append(self._create_document(file_path))
                self.logger.debug(f"문서 로드 완료: {file_path.name}")
            except Exception as e:
                self.logger.error(f"파일 로딩 실패 {file_path}: {str(e)}")
        
        self.logger.info(f"총 {len(documents)}개의 문서를 로드했습니다.")
        return documents
```

## 2. 데이터 정규화 및 전처리

### 문서 구조 분석

- 고객 리뷰 형식 인식 (별점, 텍스트, 날짜 등)
- 반정형 데이터에서 주요 필드 추출
- 정규 표현식 기반 패턴 인식

### 커스텀 파서 개발

```python
def parse_review_structure(raw_text):
    """
    고객 리뷰 텍스트에서 구조화된 정보 추출
    
    Args:
        raw_text (str): 원본 리뷰 텍스트
        
    Returns:
        dict: 구조화된 리뷰 정보
    """
    # 정규 표현식 패턴
    rating_pattern = r'평점:\s*(\d+\.?\d*)'
    date_pattern = r'날짜:\s*(\d{4}-\d{2}-\d{2})'
    
    # 정보 추출
    rating_match = re.search(rating_pattern, raw_text)
    date_match = re.search(date_pattern, raw_text)
    
    # 본문 내용 정제
    content = re.sub(r'평점:.*?\n', '', raw_text)
    content = re.sub(r'날짜:.*?\n', '', content).strip()
    
    return {
        'rating': float(rating_match.group(1)) if rating_match else None,
        'date': date_match.group(1) if date_match else None,
        'content': content
    }
```

### 정규화 프로세스

- 언어 감지 및 정규화
- 중복 제거
- 텍스트 정제 (HTML 태그, 특수문자 제거)
- 토큰화 및 문장 경계 감지

## 3. 문서 표현 및 저장

### Document 객체 모델

```python
from langchain_core.documents import Document

def _create_document(self, file_path):
    """
    파일에서 Document 객체 생성

    Args:
        file_path (Path): 파일 경로

    Returns:
        Document: 생성된 Document 객체
    """
    content = file_path.read_text(encoding='utf-8')
    doc_id = file_path.stem.split("_")[-1]
    
    # 문서 구조 분석
    parsed_content = parse_review_structure(content)
    
    # 메타데이터 강화
    metadata = {
        "source": file_path.name, 
        "id": doc_id,
        "rating": parsed_content.get('rating'),
        "date": parsed_content.get('date'),
        "file_size": file_path.stat().st_size,
        "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
    }
    
    return Document(
        page_content=parsed_content['content'],
        metadata=metadata
    )
```

## 4. 데이터 파이프라인 구축

### 파이프라인 컴포넌트

```
데이터 수집 → 전처리 → 정규화 → 인덱싱 → 분석 → 평가 → 시각화
```

### Airflow를 활용한 자동화

```python
# airflow DAG 예시
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'document_analysis_pipeline',
    default_args=default_args,
    description='고객 리뷰 문서 분석 파이프라인',
    schedule_interval=timedelta(days=1),
) as dag:
    
    # 데이터 수집 단계
    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=collect_new_data,
        op_kwargs={'target_dir': 'review_data'},
    )
    
    # 데이터 전처리 단계
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_documents,
    )
    
    # 검색 인덱스 업데이트
    update_index = PythonOperator(
        task_id='update_search_index',
        python_callable=update_search_index,
    )
    
    # 인사이트 분석 수행
    analyze_insights = PythonOperator(
        task_id='analyze_insights',
        python_callable=generate_insights,
    )
    
    # 작업 순서 정의
    collect_data >> preprocess_data >> update_index >> analyze_insights
```


## 5. 데이터 샘플 생성 프로세스

### 실습용 샘플 데이터 생성

```python
# generate_review_data.py 핵심 기능
def generate_sample_reviews(count=100, output_dir="review_data"):
    """
    학습용 리뷰 데이터 생성
    
    Args:
        count (int): 생성할 리뷰 수
        output_dir (str): 저장할 디렉토리
    """
    # 카테고리별 실제 분포 계산
    category_counts = {
        "불만 사항": int(count * 0.53),
        "개선 요구": int(count * 0.27),
        "기술적 문제": int(count * 0.165),
        "장점": int(count * 0.035)
    }
    
    # 부족한 수 보정
    remaining = count - sum(category_counts.values())
    category_counts["불만 사항"] += remaining
    
    # 데이터 생성
    reviews = []
    for category, cat_count in category_counts.items():
        for i in range(cat_count):
            review = generate_review_for_category(category)
            reviews.append(review)
    
    # 랜덤 셔플
    random.shuffle(reviews)
    
    # 파일로 저장
    os.makedirs(output_dir, exist_ok=True)
    for i, review in enumerate(reviews):
        with open(f"{output_dir}/customer_feedback_{i+1:03d}.txt", "w") as f:
            f.write(f"평점: {review['rating']}\n")
            f.write(f"날짜: {review['date']}\n\n")
            f.write(review['content'])
```


## 6. 실습: 데이터 파이프라인 구축

### 실습 1: 다양한 형식의 문서 로드
1. TXT, CSV, JSON 파일 로드 구현
2. 웹 페이지 크롤링 및 콘텐츠 추출
3. 통합 로더 클래스 개발

### 실습 2: 문서 전처리 파이프라인 구현
1. 정규화 절차 개발
2. 메타데이터 추출 및 강화
3. 전처리 성능 평가

### 실습 3: Airflow DAG 설계
1. 일별 데이터 수집 DAG 설계
2. 파이프라인 컴포넌트 구현
3. 파이프라인 모니터링 대시보드 구축


## 8. 마무리 및 다음 단계

### 학습 내용 정리
- 다양한 형식의 문서 처리 방법
- 문서 정규화 전략
- Airflow를 활용한 파이프라인 자동화

### 다음 모듈 연계
- 모듈 3에서는 정제된 문서를 활용한 LLM 기반 인사이트 생성 학습
- 데이터 파이프라인과 인사이트 생성 모듈의 통합 방법

---

## 참고 자료
- [Apache Airflow 문서](https://airflow.apache.org/docs/)
- [정규 표현식 가이드](https://docs.python.org/3/library/re.html)
