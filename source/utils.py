"""
Utility functions for the RAG system.
"""

import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import os # Added import for os.getenv in validate_environment

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging with file and console handlers"""
    logger = logging.getLogger('rag_system')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        log_dir / f"rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def generate_document_id(file_path: Path, chunk_index: int) -> str:
    """Generate unique document ID from file path and chunk index"""
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    return f"{file_path.stem}_{path_hash}_chunk_{chunk_index:03d}"


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments with intelligent boundary detection"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a natural boundary
        if end < len(text):
            # Look for sentence boundaries
            boundary_chars = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
            best_boundary = -1
            
            for boundary in boundary_chars:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos > best_boundary:
                    best_boundary = boundary_pos + len(boundary)
            
            # If no sentence boundary found, look for paragraph breaks
            if best_boundary == -1:
                boundary_pos = text.rfind('\n\n', start, end)
                if boundary_pos != -1:
                    best_boundary = boundary_pos + 2
            
            # If still no boundary, look for any newline
            if best_boundary == -1:
                boundary_pos = text.rfind('\n', start, end)
                if boundary_pos != -1:
                    best_boundary = boundary_pos + 1
            
            # Use boundary if found, otherwise use hard cut
            if best_boundary != -1:
                end = best_boundary
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
    
    return chunks


def read_file_with_encoding(file_path: Path) -> str:
    """Read file content with multiple encoding attempts"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # If all encodings fail, try binary read with error handling
    try:
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        raise IOError(f"Failed to read {file_path}: {e}")


def validate_environment() -> bool:
    """Validate required environment variables"""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    return True


def format_timestamp() -> str:
    """Generate timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if necessary"""
    path.mkdir(parents=True, exist_ok=True)
    return path
