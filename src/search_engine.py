"""
하이브리드 검색 엔진을 구현하는 모듈
키워드 기반 검색(BM25)과 의미 기반 검색(FAISS)을 결합
"""
from typing import List
import logging
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import (
    BM25_TOP_K,
    FAISS_TOP_K,
    RETRIEVER_WEIGHTS,
    LOG_FORMAT,
    LOG_LEVEL
)

class HybridSearchEngine:
    """키워드 검색과 의미 검색을 결합한 하이브리드 검색 엔진"""

    def __init__(self, documents: List[Document]):
        """
        Args:
            documents (List[Document]): 검색 대상 문서 리스트
        """
        self._setup_logging()
        self.documents = documents
        self.logger.info("하이브리드 검색 엔진 초기화 시작")
        
        self.bm25_retriever = self._create_bm25_retriever()
        self.faiss_retriever = self._create_faiss_retriever()
        self.ensemble_retriever = self._create_ensemble_retriever()
        
        self.logger.info("하이브리드 검색 엔진 초기화 완료")

    def _setup_logging(self) -> None:
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(LOG_LEVEL)

    def _create_bm25_retriever(self) -> BM25Retriever:
        """
        BM25 검색기 생성

        Returns:
            BM25Retriever: 생성된 BM25 검색기
        """
        self.logger.debug("BM25 검색기 초기화")
        retriever = BM25Retriever.from_documents(self.documents)
        retriever.k = BM25_TOP_K
        return retriever

    def _create_faiss_retriever(self):
        """
        FAISS 검색기 생성

        Returns:
            BaseRetriever: 생성된 FAISS 검색기
        """
        self.logger.debug("FAISS 검색기 초기화")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(self.documents, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": FAISS_TOP_K})

    def _create_ensemble_retriever(self) -> EnsembleRetriever:
        """
        앙상블 검색기 생성

        Returns:
            EnsembleRetriever: 생성된 앙상블 검색기
        """
        self.logger.debug("앙상블 검색기 초기화")
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[RETRIEVER_WEIGHTS["bm25"], RETRIEVER_WEIGHTS["faiss"]]
        )

    def search(self, query: str) -> List[Document]:
        """
        주어진 쿼리로 문서 검색

        Args:
            query (str): 검색 쿼리

        Returns:
            List[Document]: 검색된 관련 문서 리스트
        """
        self.logger.info(f"검색 쿼리 실행: {query}")
        results = self.ensemble_retriever.invoke(query)
        self.logger.info(f"검색 결과: {len(results)}개 문서")
        return results