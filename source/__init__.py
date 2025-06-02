"""
Modular RAG System
A production-ready Retrieval Augmented Generation system with file processing capabilities.
"""

__version__ = "1.0.0"
__author__ = "RAG Development Team"

from .config import RAGConfig, FileType
from .rag_system import ProductionRAGSystem
from .file_processor import FileProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .query_engine import QueryEngine
from .output_manager import OutputManager
from .verbatim_extractor import VerbatimExtractor, VerbatimFormat

__all__ = [
    'RAGConfig',
    'FileType',
    'ProductionRAGSystem',
    'FileProcessor',
    'EmbeddingManager',
    'VectorStore',
    'QueryEngine',
    'OutputManager',
    'VerbatimExtractor',
    'VerbatimFormat'
]
