"""
Configuration management for the RAG system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
import os


class FileType(Enum):
    """Supported file types for processing"""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    OTHER = "other"


@dataclass
class RAGConfig:
    """Comprehensive configuration for the RAG system"""
    
    # Pinecone Configuration
    index_name: str = "retrieval-test"
    
    # OpenAI Configuration
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    dimension: int = 1024
    
    # Processing Configuration
    similarity_threshold: float = 0.1
    max_retries: int = 3
    retry_delay: float = 2.0
    indexing_wait_time: int = 15
    batch_size: int = 100
    max_context_docs: int = 10
    
    # Generation Configuration
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # File Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Supported file extensions
    supported_extensions: Dict[str, FileType] = None
    
    def __post_init__(self):
        """Initialize supported extensions if not provided"""
        if self.supported_extensions is None:
            self.supported_extensions = {
                '.txt': FileType.TEXT,
                '.md': FileType.MARKDOWN,
                '.markdown': FileType.MARKDOWN,
                '.py': FileType.CODE,
                '.js': FileType.CODE,
                '.java': FileType.CODE,
                '.cpp': FileType.CODE,
                '.c': FileType.CODE,
                '.h': FileType.CODE,
                '.css': FileType.CODE,
                '.html': FileType.CODE,
                '.xml': FileType.CODE,
                '.json': FileType.JSON,
                '.yml': FileType.CODE,
                '.yaml': FileType.CODE,
                '.sql': FileType.CODE,
                '.sh': FileType.CODE,
                '.bat': FileType.CODE,
                '.csv': FileType.TEXT,
                '.log': FileType.TEXT,
                '.conf': FileType.TEXT,
                '.cfg': FileType.TEXT,
            }
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        return cls(
            index_name=os.getenv('PINECONE_INDEX_NAME', 'retrieval-test'),
            embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
            chat_model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o'),
            dimension=int(os.getenv('EMBEDDING_DIMENSION', '1024')),
            similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', '0.1')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            max_context_docs=int(os.getenv('MAX_CONTEXT_DOCS', '10'))
        )
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.max_context_docs <= 0:
            raise ValueError("Max context docs must be positive")
        return True