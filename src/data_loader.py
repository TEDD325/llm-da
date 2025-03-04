"""
문서 로딩을 담당하는 모듈
"""
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import logging
from config import LOG_FORMAT, LOG_LEVEL

class DocumentLoader:
    """문서 로딩을 처리하는 클래스"""
    
    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir (Path): 문서가 저장된 기본 디렉토리 경로
        """
        self.base_dir = base_dir
        self._setup_logging()

    def _setup_logging(self) -> None:
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(LOG_LEVEL)

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        """
        지정된 디렉토리에서 문서를 로드

        Args:
            directory (Optional[str]): 로드할 디렉토리 경로. None이면 base_dir 사용

        Returns:
            List[Document]: 로드된 문서 리스트

        Raises:
            FileNotFoundError: 디렉토리를 찾을 수 없는 경우
        """
        target_dir = Path(directory) if directory else self.base_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target_dir}")
        
        documents = []
        for file_path in target_dir.glob("*.txt"):
            try:
                documents.append(self._create_document(file_path))
                self.logger.debug(f"문서 로드 완료: {file_path.name}")
            except Exception as e:
                self.logger.error(f"파일 로딩 실패 {file_path}: {str(e)}")
        
        self.logger.info(f"총 {len(documents)}개의 문서를 로드했습니다.")
        return documents

    def _create_document(self, file_path: Path) -> Document:
        """
        파일에서 Document 객체 생성

        Args:
            file_path (Path): 파일 경로

        Returns:
            Document: 생성된 Document 객체
        """
        content = file_path.read_text(encoding='utf-8')
        doc_id = file_path.stem.split("_")[-1]
        return Document(
            page_content=content,
            metadata={"source": file_path.name, "id": doc_id}
        )