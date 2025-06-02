"""
Embedding management module for the RAG system.
Handles OpenAI API interactions for creating embeddings.
"""

import logging
import time
from typing import List
import openai
import os # Added import for os.getenv

from .config import RAGConfig


class EmbeddingManager:
    """Manage OpenAI embedding operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger('rag_system.embeddings')
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client and test connection"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _test_connection(self):
        """Test OpenAI connection with a simple embedding request"""
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=["connection test"],
                dimensions=self.config.dimension
            )
            self.logger.info("OpenAI connection test successful")
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Creating embeddings for {len(texts)} texts (attempt {attempt + 1})")
                
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=texts,
                    dimensions=self.config.dimension
                )
                
                embeddings = [embedding.embedding for embedding in response.data]
                
                # Validate embeddings
                self._validate_embeddings(embeddings, texts)
                
                self.logger.debug(f"Successfully created {len(embeddings)} embeddings")
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"Embedding creation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("All embedding creation attempts failed")
                    raise
    
    def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        embeddings = self.create_embeddings([text])
        return embeddings[0]
    
    def _validate_embeddings(self, embeddings: List[List[float]], texts: List[str]):
        """Validate embedding dimensions and content"""
        if len(embeddings) != len(texts):
            raise ValueError(f"Embedding count {len(embeddings)} doesn't match text count {len(texts)}")
        
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.config.dimension:
                raise ValueError(f"Embedding {i} has wrong dimension: {len(embedding)} vs {self.config.dimension}")
            
            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError(f"Embedding {i} contains non-numeric values")
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding configuration"""
        return {
            'model': self.config.embedding_model,
            'dimension': self.config.dimension,
            'max_retries': self.config.max_retries,
            'retry_delay': self.config.retry_delay
        }