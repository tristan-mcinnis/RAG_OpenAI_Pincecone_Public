"""
Vector store management module for the RAG system.
Handles Pinecone operations for indexing and searching.
"""

import logging
import time
from typing import List, Dict, Tuple
from pinecone import Pinecone
import os # Added import for os.getenv

from .config import RAGConfig
from .file_processor import ProcessedDocument


class VectorStore:
    """Manage Pinecone vector database operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger('rag_system.vector_store')
        self._initialize_client()
        self._connect_to_index()
    
    def _initialize_client(self):
        """Initialize Pinecone client"""
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            self.client = Pinecone(api_key=api_key)
            self.logger.info("Pinecone client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
    
    def _connect_to_index(self):
        """Connect to Pinecone index and verify configuration"""
        try:
            # FIXED: Properly get list of index names
            indexes_list = self.client.list_indexes()
            existing_indexes = [index.name for index in indexes_list]
            
            if self.config.index_name not in existing_indexes:
                raise ValueError(f"Index {self.config.index_name} does not exist. Available indexes: {existing_indexes}")
            
            self.index = self.client.Index(self.config.index_name)
            self.logger.info(f"Connected to index: {self.config.index_name}")
            
            # Verify configuration
            stats = self.index.describe_index_stats()
            if stats['dimension'] != self.config.dimension:
                raise ValueError(f"Index dimension mismatch: {stats['dimension']} vs {self.config.dimension}")
            
            self.logger.info("Index configuration verified")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to index: {e}")
            raise
    
    def add_documents(self, documents: List[ProcessedDocument], embeddings: List[List[float]]) -> bool:
        """Add documents with embeddings to the index"""
        try:
            if len(documents) != len(embeddings):
                raise ValueError(f"Document count {len(documents)} doesn't match embedding count {len(embeddings)}")
            
            # Get initial stats
            initial_stats = self.index.describe_index_stats()
            initial_count = initial_stats.get('total_vector_count', 0)
            
            # Prepare vectors for upsert
            vectors = []
            for doc, embedding in zip(documents, embeddings):
                # Store text in metadata where retrieval expects it
                metadata = doc.metadata.copy()
                metadata['text'] = doc.text
                
                vector = {
                    'id': doc.id,
                    'values': embedding,
                    'metadata': metadata
                }
                vectors.append(vector)
            
            # Upsert in batches
            self._upsert_vectors_batched(vectors)
            
            # Wait for indexing
            self.logger.info("Waiting for indexing to complete...")
            time.sleep(self.config.indexing_wait_time)
            
            # Verify indexing
            final_stats = self.index.describe_index_stats()
            final_count = final_stats.get('total_vector_count', 0)
            
            if final_count >= initial_count + len(documents):
                self.logger.info(f"Indexing verified: {final_count} total vectors")
                return True
            else:
                self.logger.warning(f"Indexing may be incomplete: expected {initial_count + len(documents)}, got {final_count}")
                return True  # May still be indexing
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to index: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """Search for similar vectors"""
        if top_k is None:
            top_k = self.config.max_context_docs
        
        try:
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Process results
            relevant_docs = []
            for match in search_results.matches:
                if match.score >= self.config.similarity_threshold:
                    # Extract text from metadata
                    text_content = match.metadata.get('text', '')
                    
                    if not text_content.strip():
                        self.logger.warning(f"Empty text content for document {match.id}")
                    
                    relevant_docs.append({
                        'id': match.id,
                        'score': match.score,
                        'text': text_content,
                        'metadata': match.metadata
                    })
            
            self.logger.info(f"Search found {len(search_results.matches)} total matches, {len(relevant_docs)} above threshold")
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def _upsert_vectors_batched(self, vectors: List[Dict]) -> bool:
        """Upsert vectors in batches with error handling"""
        try:
            total_batches = (len(vectors) + self.config.batch_size - 1) // self.config.batch_size
            self.logger.info(f"Upserting {len(vectors)} vectors in {total_batches} batches")
            
            for i in range(0, len(vectors), self.config.batch_size):
                batch = vectors[i:i + self.config.batch_size]
                batch_num = i // self.config.batch_size + 1
                
                # Retry batch upsert
                for attempt in range(self.config.max_retries):
                    try:
                        result = self.index.upsert(vectors=batch)
                        self.logger.debug(f"Batch {batch_num}/{total_batches} upserted successfully")
                        break
                    except Exception as e:
                        self.logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {e}")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                        else:
                            self.logger.error(f"Batch {batch_num} failed after all attempts")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch upsert failed: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get current index statistics"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_all_vectors(self) -> bool:
        """Delete all vectors from the index (use with caution)"""
        try:
            self.index.delete(delete_all=True)
            self.logger.warning("All vectors deleted from index")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete all vectors: {e}")
            return False
