# 모듈 4: 평가 및 시각화

## 학습 목표
- LLM 기반 문서 분석 평가를 위한 핵심 지표 이해
- 분포 인식 평가 구현 방법 학습
- 인사이트 분석을 위한 시각화 기법 습득
- 시스템 성능 모니터링 대시보드 개발

## 1. 분포 인식 평가

### 1.1 카테고리 분포 지표
- 목표 분포 준수 측정
  - 불만 사항: 53%
  - 개선 요구: 27%
  - 기술적 문제: 16.5%
  - 장점: 3.5%
- 분포 추적 구현
- 분포 드리프트 계산

### 1.2 품질 지표
- 신뢰도 점수 분석
- 의미적 관련성 평가
- 카테고리 분류 정확도
- 카테고리 간 인사이트 다양성

## 2. 시각화 기법

### 2.1 분포 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_category_distribution(actual_dist, generated_dist):
    categories = ['불만', '개선', '기술', '장점']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 실제 분포
    ax1.pie(actual_dist, labels=categories, autopct='%1.1f%%')
    ax1.set_title('실제 분포')
    
    # 생성된 분포
    ax2.pie(generated_dist, labels=categories, autopct='%1.1f%%')
    ax2.set_title('생성된 분포')
    
    plt.tight_layout()
    return fig
```

### 2.2 성능 모니터링
```python
def plot_performance_metrics(metrics_over_time):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 신뢰도 점수
    sns.lineplot(data=metrics_over_time, x='timestamp', y='confidence_score', ax=axes[0,0])
    axes[0,0].set_title('시간에 따른 신뢰도 점수')
    
    # 분포 준수도
    sns.lineplot(data=metrics_over_time, x='timestamp', y='distribution_error', ax=axes[0,1])
    axes[0,1].set_title('시간에 따른 분포 오차')
    
    # 처리 시간
    sns.lineplot(data=metrics_over_time, x='timestamp', y='processing_time', ax=axes[1,0])
    axes[1,0].set_title('처리 시간')
    
    # 카테고리 정확도
    sns.lineplot(data=metrics_over_time, x='timestamp', y='category_accuracy', ax=axes[1,1])
    axes[1,1].set_title('카테고리 분류 정확도')
    
    plt.tight_layout()
    return fig
```

## 3. 모니터링 대시보드

### 3.1 실시간 모니터링
- Streamlit 대시보드 설정
- 실시간 업데이트 구현
- 경고 임계값 구성

```python
import streamlit as st

def create_monitoring_dashboard():
    st.title("문서 분석 모니터링")
    
    # 분포 지표
    st.header("카테고리 분포")
    distribution_fig = plot_category_distribution(actual_dist, generated_dist)
    st.pyplot(distribution_fig)
    
    # 성능 지표
    st.header("시스템 성능")
    performance_fig = plot_performance_metrics(metrics_df)
    st.pyplot(performance_fig)
    
    # 경고 설정
    st.sidebar.header("경고 설정")
    confidence_threshold = st.sidebar.slider("신뢰도 점수 임계값", 0.0, 1.0, 0.8)
    distribution_error_threshold = st.sidebar.slider("분포 오차 임계값", 0.0, 1.0, 0.1)
```

### 3.2 경고 시스템
- 신뢰도 점수 경고
- 분포 드리프트 감지
- 처리 시간 모니터링
- 오류율 추적

## 4. 시스템 최적화

### 4.1 성능 튜닝
- 캐싱 전략
- 배치 처리 최적화
- 리소스 사용률 모니터링

### 4.2 품질 개선
- 지표 기반 프롬프트 개선
- 분포 조정 기법
- 신뢰도 점수 보정

## 실습 과제
1. 시각화 함수 구현
2. 기본 모니터링 대시보드 설정
3. 경고 임계값 구성
4. 제공된 지표를 사용하여 시스템 성능 분석

## 추가 자료
- Matplotlib 문서
- Streamlit 튜토리얼
- 모니터링 모범 사례
- 성능 최적화 가이드

## 평가
1. 인사이트 분석을 위한 맞춤형 시각화 만들기
2. 실시간 모니터링 대시보드 구현
3. 경고 시스템 구성 및 테스트
4. 지표를 기반으로 시스템 성능 최적화
