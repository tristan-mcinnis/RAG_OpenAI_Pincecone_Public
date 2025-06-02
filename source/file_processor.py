"""
File processing module for the RAG system.
Handles file discovery, reading, and chunking.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from .config import RAGConfig, FileType
from .utils import generate_document_id, chunk_text, read_file_with_encoding


@dataclass
class ProcessedDocument:
    """Container for processed document information"""
    id: str
    text: str
    source_file: str
    file_type: FileType
    chunk_index: int
    total_chunks: int
    file_size: int
    metadata: Dict


class FileProcessor:
    """Handle file discovery, reading, and text extraction"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger('rag_system.file_processor')
    
    def discover_files(self, path: str) -> List[Path]:
        """Discover all supported text files in path (file or directory)"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if path_obj.is_file():
            if self._is_supported_file(path_obj):
                return [path_obj]
            else:
                raise ValueError(f"Unsupported file type: {path_obj.suffix}")
        
        elif path_obj.is_dir():
            files = []
            for ext in self.config.supported_extensions.keys():
                files.extend(path_obj.rglob(f"*{ext}"))
            
            # Sort for consistent processing order
            files.sort()
            self.logger.info(f"Discovered {len(files)} supported files in {path}")
            return files
        
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")
    
    def process_file(self, file_path: Path) -> List[ProcessedDocument]:
        """Process a single file into document chunks"""
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.config.max_file_size:
                self.logger.warning(f"File {file_path} exceeds size limit ({file_size} bytes)")
                return []
            
            # Read file content
            content = read_file_with_encoding(file_path)
            if not content.strip():
                self.logger.warning(f"File {file_path} is empty or contains only whitespace")
                return []
            
            # Determine file type
            file_type = self.config.supported_extensions.get(
                file_path.suffix.lower(), 
                FileType.OTHER
            )
            
            # Chunk the content
            chunks = chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
            self.logger.info(f"File {file_path.name} chunked into {len(chunks)} pieces")
            
            # Create processed documents
            documents = []
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    self.logger.warning(f"Empty chunk {i} in file {file_path.name}")
                    continue
                    
                doc_id = generate_document_id(file_path, i)
                
                doc = ProcessedDocument(
                    id=doc_id,
                    text=chunk,
                    source_file=str(file_path),
                    file_type=file_type,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    file_size=file_size,
                    metadata={
                        'source_file': str(file_path),
                        'file_name': file_path.name,
                        'file_extension': file_path.suffix,
                        'file_type': file_type.value,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'file_size': file_size,
                        'directory': str(file_path.parent),
                        'processed_timestamp': time.time(),
                        'chunk_length': len(chunk)
                    }
                )
                documents.append(doc)
            
            self.logger.info(f"Processed {file_path} into {len(documents)} valid chunks")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_multiple_files(self, file_paths: List[Path]) -> List[ProcessedDocument]:
        """Process multiple files and return all processed documents"""
        all_documents = []
        failed_files = 0
        
        for file_path in file_paths:
            docs = self.process_file(file_path)
            if docs:
                all_documents.extend(docs)
            else:
                failed_files += 1
        
        self.logger.info(f"Processed {len(all_documents)} total chunks from {len(file_paths)} files ({failed_files} failed)")
        return all_documents
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file extension is supported"""
        return file_path.suffix.lower() in self.config.supported_extensions