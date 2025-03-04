"""
LLM 기반 통찰 분석을 수행하는 모듈
실제 데이터 분포를 반영한 통찰 생성
"""
from typing import List, Dict, Any
import json
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import (
    MODEL_NAME,
    REAL_DISTRIBUTION,
    LOG_FORMAT,
    LOG_LEVEL
)

class InsightAnalyzer:
    """LLM을 사용하여 문서에서 통찰을 도출하는 클래스"""

    def __init__(self):
        """분석기 초기화"""
        self._setup_logging()
        self.logger.info("통찰 분석기 초기화")
        self.llm = ChatOpenAI(model=MODEL_NAME)
        self.prompt = self._create_analysis_prompt()

    def _setup_logging(self) -> None:
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(LOG_LEVEL)

    def analyze(self, documents: List[Document], query: str) -> List[Dict[str, Any]]:
        """
        문서 분석을 수행하고 통찰 도출

        Args:
            documents (List[Document]): 분석할 문서 리스트
            query (str): 분석 쿼리

        Returns:
            List[Dict[str, Any]]: 도출된 통찰 리스트

        Raises:
            ValueError: LLM 응답 처리 중 오류 발생 시
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
          {{
            "insight_type": "개선 요구",
            "insight_value": "구체적인 통찰 내용",
            "confidence_score": 0.85
          }},
          {{
            "insight_type": "기술적 문제",
            "insight_value": "구체적인 통찰 내용",
            "confidence_score": 0.75
          }},
          {{
            "insight_type": "장점",
            "insight_value": "구체적인 통찰 내용",
            "confidence_score": 0.65
          }}
        ]
        ```

        응답 시 다음 사항을 반드시 준수해주세요:
        1. 모든 카테고리(불만 사항, 개선 요구, 기술적 문제, 장점)에 대해 각각 최소 1개 이상의 통찰 제공 (누락된 카테고리가 없어야 함)
        2. 전체 통찰 개수는 최소 6개 이상
        3. 실제 데이터 분포 비율에 맞게 각 카테고리의 통찰 수 조정
        4. confidence_score는 0.0부터 1.0 사이의 실수
        5. 비율이 매우 낮은 '장점' 카테고리도 반드시 포함해야 함
        6. insight_type은 정확히 다음 값 중 하나여야 함: "불만 사항", "개선 요구", "기술적 문제", "장점"
        """
        return ChatPromptTemplate.from_template(template)

    def _generate_insights(self, context: str, query: str) -> str:
        """
        LLM을 사용하여 통찰 생성

        Args:
            context (str): 분석할 컨텍스트
            query (str): 분석 쿼리

        Returns:
            str: LLM 응답
        """
        self.logger.debug("LLM 통찰 생성 시작")
        result = self.prompt | self.llm
        return result.invoke({
            "context": context,
            "question": query
        })

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
            if not 0 <= insight.get("confidence_score", -1) <= 1:
                raise ValueError("신뢰도 점수가 유효하지 않습니다")