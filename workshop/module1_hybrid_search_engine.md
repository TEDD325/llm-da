# 모듈 1: LLM 기반 하이브리드 검색 엔진

## 학습 목표
- 키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)의 작동 원리 이해
- 하이브리드 검색 시스템 구현 방법 습득
- OpenAI 임베딩을 활용한 벡터 검색 최적화 기법 학습

## 1. 하이브리드 검색의 필요성

### 키워드 검색과 의미 검색의 한계

**키워드 검색(BM25)의 한계:**
- 동의어 인식 불가 (예: "불만" vs "불평")
- 맥락 이해 부족 ("시스템이 느리다" vs "응답 속도가 늦다")
- 관련 문서이지만 정확한 키워드가 없으면 검색 실패

**의미 검색(FAISS)의 한계:**
- 정확한 키워드 매칭에 약함
- 계산 비용이 높음
- 임베딩 모델의 품질에 의존적

### 하이브리드 접근법의 이점

```python
# 단일 방식 대비 하이브리드 방식의 성능 향상 사례
# Recall@5 기준 (동일 데이터셋)
performance_comparison = {
    "BM25만 사용": "67%",
    "FAISS만 사용": "73%",
    "하이브리드 방식": "89%"  # 두 방식의 장점을 결합
}
```

## 2. BM25 검색기 구현

### BM25 알고리즘 이해

BM25는 TF-IDF의 개선된 버전으로, 다음 요소를 고려합니다:
- 단어 빈도(TF) 포화 함수
- 문서 길이 정규화
- 역문서 빈도(IDF) 가중치

### 코드 구현

```python
from langchain_community.retrievers import BM25Retriever

def create_bm25_retriever(documents, top_k=3):
    """BM25 검색기 생성"""
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k
    return retriever
    
# 사용 예시
bm25_results = bm25_retriever.invoke("고객이 제품 배송에 불만을 표시했습니다")
```

**실습**: BM25 검색에서 다양한 `k` 값 실험과 그 영향 분석

## 3. 벡터 검색(FAISS) 구현

### 문서 임베딩의 원리

텍스트를 고차원 벡터 공간에 매핑하는 방법:
- 토큰화 → 임베딩 변환 → 벡터 정규화
- OpenAI 임베딩 vs 오픈소스 대안(Sentence-BERT)

### FAISS를 활용한 벡터 검색 구현

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_faiss_retriever(documents, top_k=3):
    """FAISS 벡터 검색기 생성"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# 사용 예시
faiss_results = faiss_retriever.invoke("소프트웨어 성능 개선 요구사항")
```

### 임베딩 최적화 기법

- 배치 처리를 통한 API 비용 절감
- 임베딩 캐싱 전략
- 문서 청킹 전략 (최적 청크 크기 결정)

## 4. 앙상블 검색기 구현

### 검색 결과 결합 전략

**가중치 기반 결합**:
```python
from langchain.retrievers import EnsembleRetriever

def create_ensemble_retriever(bm25_retriever, faiss_retriever, weights=None):
    """앙상블 검색기 생성"""
    if weights is None:
        weights = [0.4, 0.6]  # BM25 40%, FAISS 60%
        
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=weights
    )
```

**가중치 최적화 실험**:
```
실험 결과 (Precision@5 기준):
- BM25 30% + FAISS 70%: 0.82
- BM25 40% + FAISS 60%: 0.87  # 최적 조합
- BM25 50% + FAISS 50%: 0.85
```

### 검색 결과 재랭킹

- MMR(Maximum Marginal Relevance)를 활용한 다양성 확보
- 문서 길이 페널티 적용
- 최신성 가중치 부여

## 5. 실제 구현: HybridSearchEngine 클래스

```python
class HybridSearchEngine:
    """키워드 검색과 의미 검색을 결합한 하이브리드 검색 엔진"""

    def __init__(self, documents):
        self._setup_logging()
        self.documents = documents
        self.logger.info("하이브리드 검색 엔진 초기화 시작")
        
        self.bm25_retriever = self._create_bm25_retriever()
        self.faiss_retriever = self._create_faiss_retriever()
        self.ensemble_retriever = self._create_ensemble_retriever()
        
        self.logger.info("하이브리드 검색 엔진 초기화 완료")

    def search(self, query):
        """주어진 쿼리로 문서 검색"""
        self.logger.info(f"검색 쿼리 실행: {query}")
        results = self.ensemble_retriever.invoke(query)
        self.logger.info(f"검색 결과: {len(results)}개 문서")
        return results
```

## 6. 실습: 하이브리드 검색 성능 평가

### 평가 지표 설정
- Precision@K: 상위 K개 결과 중 관련 문서의 비율
- Recall@K: 관련 문서 중 검색된 비율
- F1 Score: Precision과 Recall의 조화 평균

### 실습 과제
1. 샘플 데이터셋에 대해 BM25만 사용한 검색, FAISS만 사용한 검색, 하이브리드 검색 결과 비교
2. 다양한 가중치 조합에 따른 검색 결과 변화 분석
3. 실제 비즈니스 시나리오에 맞는 최적 가중치 도출


### 캐싱 전략
```python
# 캐싱 인프라 구현 예시
from functools import lru_cache
import hashlib
from datetime import datetime, timedelta

class SearchCacheManager:
    def __init__(self, ttl_hours=1):
        self.cache = {}
        self.cache_ttl = timedelta(hours=ttl_hours)
    
    def get_cache_key(self, query, params):
        """검색 쿼리와 파라미터로 캐시 키 생성"""
        key_str = f"{query}_{str(params)}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get_cached_result(self, key):
        """캐시된 결과 조회 (TTL 체크)"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expiry']:
                return entry['result']
        return None
        
    def cache_result(self, key, result):
        """검색 결과 캐싱"""
        self.cache[key] = {
            'result': result,
            'expiry': datetime.now() + self.cache_ttl
        }
```

## 8. 마무리 및 다음 단계

### 학습 내용 정리
- 하이브리드 검색의 원리와 이점
- BM25와 FAISS 구현 방법
- 앙상블 검색기 최적화 전략

### 다음 모듈 연계
- 모듈 2에서는 다양한 형식의 문서를 로드하고 전처리하는 방법 학습
- 검색 결과를 LLM 분석기에 전달하는 파이프라인 구축

---

## 참고 자료
- [FAISS 공식 문서](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI 임베딩 API 가이드](https://platform.openai.com/docs/guides/embeddings)
