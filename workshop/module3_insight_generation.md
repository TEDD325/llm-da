# 모듈 3: LLM 기반 인사이트 생성

## 학습 목표
- LLM을 활용한 문서 분석 및 인사이트 추출 방법 이해
- 실제 데이터 분포를 반영한 인사이트 생성 기법 습득
- 프롬프트 엔지니어링 최적화 전략 학습

## 1. LLM 기반 텍스트 분석의 기초

### LLM의 문서 이해 능력

**LLM의 강점:**
- 자연어 이해 및 맥락 파악
- 다양한 형식과 스타일의 텍스트 처리
- 암시적 정보 추론
- 요약 및 핵심 포인트 추출

**전통적 NLP 기법과의 비교:**
```
| 기능 | 전통적 NLP | LLM 기반 분석 |
|------|------------|--------------|
| 감성 분석 | 사전 기반, 규칙 기반 | 맥락 이해, 뉘앙스 포착 |
| 주제 추출 | TF-IDF, LDA | 의미론적 주제 파악 |
| 개체명 인식 | 패턴 매칭, CRF | 맥락 기반 개체 인식 |
| 관계 추출 | 구문 분석 기반 | 암시적 관계 파악 |
```

### InsightAnalyzer 클래스 개요

```python
class InsightAnalyzer:
    """LLM을 사용하여 문서에서 통찰을 도출하는 클래스"""

    def __init__(self):
        """분석기 초기화"""
        self._setup_logging()
        self.logger.info("통찰 분석기 초기화")
        self.llm = ChatOpenAI(model=MODEL_NAME)
        self.prompt = self._create_analysis_prompt()

    def analyze(self, documents: List[Document], query: str) -> List[Dict[str, Any]]:
        """
        문서 분석을 수행하고 통찰 도출

        Args:
            documents (List[Document]): 분석할 문서 리스트
            query (str): 분석 쿼리

        Returns:
            List[Dict[str, Any]]: 도출된 통찰 리스트
        """
        try:
            self.logger.info("문서 분석 시작")
            context = "\n".join([doc.page_content for doc in documents])
            result = self._generate_insights(context, query)
            insights = self._process_llm_response(result)
            self.logger.info(f"분석 완료: {len(insights)}개의 통찰 도출")
            return insights
        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {str(e)}")
            raise
```

## 2. 프롬프트 엔지니어링 최적화

### 효과적인 분석 프롬프트 설계

**프롬프트 구성 요소:**
1. 작업 정의 (Task Definition)
2. 컨텍스트 제공 (Context Provision)
3. 출력 형식 지정 (Output Formatting)
4. 제약 조건 설정 (Constraints)
5. 예시 제공 (Few-shot Examples)

```python
def _create_analysis_prompt(self) -> ChatPromptTemplate:
    """
    분석용 프롬프트 템플릿 생성

    Returns:
        ChatPromptTemplate: 생성된 프롬프트 템플릿
    """
    template = """
    다음 데이터에서 패턴을 분석하고 통찰을 도출해주세요:
    {context}

    분석 요청: {question}

    실제 데이터 분포 정보:
    """
    for category, percentage in REAL_DISTRIBUTION.items():
        template += f"- {category}: 약 {percentage}%\n"
    
    template += """
    위의 실제 데이터 분포를 참고하여, JSON 배열로 응답해주세요:
    ```json
    [
      {{
        "insight_type": "불만 사항",
        "insight_value": "구체적인 통찰 내용",
        "confidence_score": 0.95
      }},
      ...
    ]
    ```

    응답 시 다음 사항을 반드시 준수해주세요:
    1. 모든 카테고리(불만 사항, 개선 요구, 기술적 문제, 장점)에 대해 각각 최소 1개 이상의 통찰 제공
    2. 전체 통찰 개수는 최소 6개 이상
    3. 실제 데이터 분포 비율에 맞게 각 카테고리의 통찰 수 조정
    4. confidence_score는 0.0부터 1.0 사이의 실수
    5. 비율이 매우 낮은 '장점' 카테고리도 반드시 포함해야 함
    """
    return ChatPromptTemplate.from_template(template)
```

### 프롬프트 최적화 전략

**명확한 지시와 제약:**
- 구체적인 작업 정의
- 출력 형식 명시
- 평가 기준 제공

**컨텍스트 최적화:**
- 관련성 높은 정보 우선 제공
- 문서 요약 또는 청킹
- 핵심 정보 강조

## 3. 실제 데이터 분포 반영 전략

### 분포 인식 샘플링 원리

**카테고리 분포 불균형 문제:**
- 실제 데이터: 불만 사항(53%), 개선 요구(27%), 기술적 문제(16.5%), 장점(3.5%)
- 균형 잡힌 인사이트 생성의 어려움
- 희소 카테고리(장점) 누락 위험

**분포 조정 알고리즘:**
```python
def adjust_distribution(insights, target_distribution):
    """
    생성된 인사이트의 분포를 목표 분포에 맞게 조정
    
    Args:
        insights (List[Dict]): 원본 인사이트 리스트
        target_distribution (Dict): 목표 카테고리 분포
        
    Returns:
        List[Dict]: 조정된 인사이트 리스트
    """
    # 현재 분포 계산
    current_counts = {}
    for cat in CATEGORIES:
        current_counts[cat] = sum(1 for i in insights if i["insight_type"] == cat)
    
    total_insights = len(insights)
    current_dist = {cat: count/total_insights*100 for cat, count in current_counts.items()}
    
    # 목표 개수 계산 (최소 6개 인사이트 기준)
    target_count = max(6, total_insights)
    target_counts = {cat: max(1, int(target_count * pct/100)) for cat, pct in target_distribution.items()}
    
    # 부족한 카테고리 식별
    missing_categories = []
    for cat in CATEGORIES:
        if current_counts[cat] < target_counts[cat]:
            missing_categories.append((cat, target_counts[cat] - current_counts[cat]))
    
    # 과잉 카테고리 식별
    excess_categories = []
    for cat in CATEGORIES:
        if current_counts[cat] > target_counts[cat]:
            excess_categories.append((cat, current_counts[cat] - target_counts[cat]))
    
    # 조정된 인사이트 리스트 구성
    adjusted_insights = insights.copy()
    
    # 부족한 카테고리 보완 (추가 생성 또는 유사 카테고리에서 변환)
    # 실제 구현에서는 LLM을 활용하여 부족한 카테고리의 인사이트 추가 생성
    
    return adjusted_insights
```

### 분포 검증 및 보정

**분포 검증 메커니즘:**
- 카테고리별 인사이트 수 계산
- 목표 분포와의 편차 측정
- 최소 표현 보장 (각 카테고리 최소 1개)

**보정 전략:**
- 부족한 카테고리 추가 생성
- 과잉 카테고리 필터링 (신뢰도 기반)
- 카테고리 변환 (유사 카테고리 활용)

## 4. LLM 응답 처리 및 검증

### JSON 파싱 및 오류 처리

```python
def _process_llm_response(self, response: str) -> List[Dict[str, Any]]:
    """
    LLM 응답을 파싱하여 통찰 리스트 생성

    Args:
        response (str): LLM 응답

    Returns:
        List[Dict[str, Any]]: 파싱된 통찰 리스트

    Raises:
        ValueError: JSON 파싱 실패 시
    """
    try:
        # LLM 응답에서 content 추출
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, dict) and 'content' in response:
            content = response['content']
        elif isinstance(response, str):
            content = response
        else:
            raise ValueError("지원되지 않는 응답 형식")
        
        # JSON 추출
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_matches = re.findall(json_pattern, content)
        
        if not json_matches:
            raise ValueError("JSON 형식의 통찰을 찾을 수 없습니다")
        
        json_str = json_matches[0].strip()
        insights = json.loads(json_str)
        
        # 결과 검증 및 변환
        if not isinstance(insights, list):
            if isinstance(insights, dict) and 'insights' in insights:
                insights = insights['insights']
            else:
                insights = [insights]
        
        self._validate_insights(insights)
        return insights
        
    except Exception as e:
        self.logger.error(f"LLM 응답 처리 중 오류: {str(e)}")
        raise ValueError(f"통찰 처리 실패: {str(e)}")
```

### 인사이트 유효성 검증

```python
def _validate_insights(self, insights: List[Dict[str, Any]]) -> None:
    """
    생성된 통찰의 유효성 검증

    Args:
        insights (List[Dict[str, Any]]): 검증할 통찰 리스트

    Raises:
        ValueError: 유효성 검증 실패 시
    """
    if len(insights) < 6:
        raise ValueError("통찰 수가 부족합니다 (최소 6개 필요)")
    
    categories_found = set(insight["insight_type"] for insight in insights)
    if len(categories_found) < len(REAL_DISTRIBUTION):
        missing = set(REAL_DISTRIBUTION.keys()) - categories_found
        raise ValueError(f"누락된 카테고리가 있습니다: {missing}")
    
    for insight in insights:
        if "insight_type" not in insight:
            raise ValueError("insight_type 필드가 누락되었습니다")
        if insight["insight_type"] not in REAL_DISTRIBUTION:
            raise ValueError(f"유효하지 않은 카테고리: {insight['insight_type']}")
        if "insight_value" not in insight:
            raise ValueError("insight_value 필드가 누락되었습니다")
        if "confidence_score" not in insight:
            raise ValueError("confidence_score 필드가 누락되었습니다")
        
        # 신뢰도 점수 검증
        score = insight["confidence_score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            raise ValueError(f"유효하지 않은 신뢰도 점수: {score}")
```

## 5. 신뢰도 점수 활용

### 신뢰도 점수의 의미

- 인사이트의 확실성 정도
- 데이터 지원 수준
- 일반화 가능성

### 신뢰도 기반 필터링 및 랭킹

```python
def filter_insights_by_confidence(insights, threshold=0.7):
    """
    신뢰도 점수 기반 인사이트 필터링
    
    Args:
        insights (List[Dict]): 인사이트 리스트
        threshold (float): 신뢰도 임계값
        
    Returns:
        List[Dict]: 필터링된 인사이트 리스트
    """
    return [insight for insight in insights if insight["confidence_score"] >= threshold]

def rank_insights_by_confidence(insights):
    """
    신뢰도 점수 기반 인사이트 랭킹
    
    Args:
        insights (List[Dict]): 인사이트 리스트
        
    Returns:
        List[Dict]: 랭킹된 인사이트 리스트
    """
    return sorted(insights, key=lambda x: x["confidence_score"], reverse=True)
```

## 6. 실습: LLM 기반 인사이트 생성

### 실습 1: 프롬프트 최적화
1. 기본 프롬프트와 최적화된 프롬프트 비교
2. 다양한 제약 조건 실험
3. 프롬프트 템플릿 개발

### 실습 2: 분포 인식 인사이트 생성
1. 실제 데이터 분포 분석
2. 분포 반영 알고리즘 구현
3. 생성 결과 평가

### 실습 3: 인사이트 검증 및 보정
1. 유효성 검사 메커니즘 구현
2. 부족한 카테고리 보완 전략
3. 신뢰도 기반 필터링 적용

## 7. 고급 주제: 인사이트 품질 향상

### 다양성 확보 전략
- 중복 인사이트 제거
- 다양한 관점 유도
- 상충되는 인사이트 균형

### 인사이트 심화 분석
- 인사이트 간 관계 분석
- 인과 관계 추론
- 시간적 추세 분석

### 멀티모달 인사이트
- 텍스트 + 이미지 분석
- 차트 데이터 해석
- 멀티모달 프롬프트 설계

## 8. 마무리 및 다음 단계

### 학습 내용 정리
- LLM 기반 문서 분석 원리
- 프롬프트 엔지니어링 최적화
- 분포 인식 인사이트 생성
- 신뢰도 평가 및 활용

### 다음 모듈 연계
- 모듈 4에서는 생성된 인사이트의 평가 및 시각화 방법 학습
- 인사이트 품질 평가 메트릭과 시각화 기법 연계

---

## 참고 자료
- [OpenAI ChatGPT API 문서](https://platform.openai.com/docs/guides/chat)
- [효과적인 프롬프트 엔지니어링 전략](https://www.promptingguide.ai/)
