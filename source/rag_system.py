"""
Main RAG system orchestration module.
Coordinates all components to provide the complete RAG functionality.
"""

import logging
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

from .config import RAGConfig
from .file_processor import FileProcessor, ProcessedDocument
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .query_engine import QueryEngine
from .output_manager import OutputManager
from .verbatim_extractor import VerbatimExtractor, VerbatimFormat
from .utils import setup_logging


class ProductionRAGSystem:
    """Main RAG system that coordinates all components"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.config.validate()
        
        # Setup logging
        self.logger = setup_logging()
        
        # Initialize components
        self.file_processor = FileProcessor(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.vector_store = VectorStore(self.config)
        self.query_engine = QueryEngine(self.config)
        self.output_manager = OutputManager()
        self.verbatim_extractor = VerbatimExtractor(self.config)  # NEW
        
        self.logger.info("Production RAG System initialized successfully")
    
    def process_and_index_files(self, file_path: str) -> Tuple[int, int]:
        """Process files and add to vector store"""
        self.logger.info(f"Starting file processing for: {file_path}")
        
        try:
            # Discover files
            files = self.file_processor.discover_files(file_path)
            self.logger.info(f"Found {len(files)} files to process")
            
            if not files:
                self.logger.warning("No supported files found")
                return 0, 0
            
            # Process all files
            all_documents: List[ProcessedDocument] = []
            failed_files = 0
            
            for p_file_path in tqdm(files, desc="Processing files"):
                docs = self.file_processor.process_file(p_file_path)
                if docs:
                    all_documents.extend(docs)
                else:
                    failed_files += 1
            
            if not all_documents:
                self.logger.error("No documents were successfully processed")
                return 0, failed_files
            
            self.logger.info(f"Processed {len(all_documents)} document chunks from {len(files)} files")
            
            # Create embeddings and index documents
            success = self._index_documents(all_documents)
            
            if success:
                self.logger.info(f"Successfully indexed {len(all_documents)} documents")
                return len(all_documents), failed_files
            else:
                self.logger.error("Failed to index documents")
                return 0, len(files)
                
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            return 0, len(files) if 'files' in locals() else 1
    
    def query_knowledge_base(self, query: str) -> Dict:
        """Query the knowledge base and generate response"""
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Validate query
            self.query_engine.validate_query(query)
            
            # Create query embedding
            query_embedding = self.embedding_manager.create_single_embedding(query)
            
            # Search for relevant documents
            relevant_docs = self.vector_store.search(query_embedding)
            
            # Generate response
            answer = self.query_engine.generate_response(query, relevant_docs)
            
            result = {
                'query': query,
                'answer': answer,
                'sources': relevant_docs,
                'total_matches': len(relevant_docs),
                'relevant_matches': len(relevant_docs),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Query processed successfully with {len(relevant_docs)} relevant sources")
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {
                'query': query,
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def extract_verbatims(
        self,
        query: str,
        min_length: int = 20,
        max_length: int = 500,
        exclude_moderator: bool = True,
        participant_filter: str = None,
        format_style: str = "research",
        top_k: int = 20
    ) -> Dict:
        """Extract verbatims from the knowledge base"""
        
        try:
            self.logger.info(f"Extracting verbatims for: {query[:50]}...")
            
            # Create query embedding and search
            query_embedding = self.embedding_manager.create_single_embedding(query)
            relevant_docs = self.vector_store.search(query_embedding, top_k=top_k)
            
            if not relevant_docs:
                return {
                    'query': query,
                    'verbatims': [],
                    'formatted_verbatims': [],
                    'total_found': 0,
                    'message': 'No relevant documents found for verbatim extraction'
                }
            
            # Convert format string to enum
            format_enum = VerbatimFormat.RESEARCH
            if format_style.lower() == "quotes_only":
                format_enum = VerbatimFormat.QUOTES_ONLY
            elif format_style.lower() == "detailed":
                format_enum = VerbatimFormat.DETAILED
            elif format_style.lower() == "csv":
                format_enum = VerbatimFormat.CSV
            
            # Extract verbatims
            verbatims = self.verbatim_extractor.extract_verbatims(
                query=query,
                retrieved_docs=relevant_docs,
                min_length=min_length,
                max_length=max_length,
                exclude_moderator=exclude_moderator,
                participant_filter=participant_filter,
                format_style=format_enum
            )
            
            # Format verbatims
            if format_enum == VerbatimFormat.CSV:
                formatted_output = self.verbatim_extractor.export_to_csv(verbatims)
            else:
                formatted_verbatims = self.verbatim_extractor.format_verbatims(verbatims, format_enum)
                formatted_output = formatted_verbatims
            
            result = {
                'query': query,
                'verbatims': [
                    {
                        'quote': v.cleaned_quote,
                        'speaker': v.speaker.name,
                        'demographics': v.speaker.demographics,
                        'location': v.speaker.location,
                        'relevance_score': v.relevance_score,
                        'word_count': v.word_count,
                        'timestamp': v.timestamp
                    } for v in verbatims
                ],
                'formatted_verbatims': formatted_output,
                'total_found': len(verbatims),
                'extraction_params': {
                    'min_length': min_length,
                    'max_length': max_length,
                    'exclude_moderator': exclude_moderator,
                    'participant_filter': participant_filter,
                    'format_style': format_style,
                    'top_k': top_k
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Extracted {len(verbatims)} verbatims")
            return result
            
        except Exception as e:
            self.logger.error(f"Verbatim extraction failed: {e}")
            return {
                'query': query,
                'verbatims': [],
                'formatted_verbatims': [],
                'total_found': 0,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def process_and_query(self, file_path: str, query: str) -> Tuple[Dict, Tuple[int, int]]:
        """Complete workflow: process files and query"""
        # Process and index files
        file_stats = self.process_and_index_files(file_path)
        
        # Query the knowledge base
        result = self.query_knowledge_base(query)
        
        return result, file_stats
    
    def _index_documents(self, documents: List[ProcessedDocument]) -> bool:
        """Create embeddings and index documents"""
        try:
            # Create embeddings in batches
            all_embeddings = []
            
            with tqdm(total=len(documents), desc="Creating embeddings and indexing") as pbar:
                for i in range(0, len(documents), self.config.batch_size):
                    batch_docs = documents[i:i + self.config.batch_size]
                    
                    # Create embeddings for batch
                    texts = [doc.text for doc in batch_docs]
                    
                    # Check for empty texts
                    empty_texts_indices = [j for j, text in enumerate(texts) if not text.strip()]
                    if empty_texts_indices:
                        self.logger.warning(f"Found {len(empty_texts_indices)} empty texts in batch {i//self.config.batch_size + 1}")
                        valid_batch_docs = [doc for idx, doc in enumerate(batch_docs) if idx not in empty_texts_indices]
                        texts = [doc.text for doc in valid_batch_docs]
                        if not texts:
                            pbar.update(len(batch_docs))
                            continue
                    else:
                        valid_batch_docs = batch_docs

                    batch_embeddings = self.embedding_manager.create_embeddings(texts)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Index batch
                    if not self.vector_store.add_documents(valid_batch_docs, batch_embeddings):
                        self.logger.error(f"Failed to index batch starting at document {i}")
                        return False
                    pbar.update(len(batch_docs))
            
            self.logger.info(f"Successfully created {len(all_embeddings)} embeddings and indexed documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Embedding and indexing failed: {e}")
            return False

    def run_interactive_session(self):
        """Run an interactive session for querying the RAG system"""
        self.logger.info("Starting interactive RAG session...")
        print("Welcome to the Interactive RAG System!")
        print("Type 'exit' or 'quit' to end the session.")

        # Initial file processing
        while True:
            file_path_input = input("Enter the path to a file or directory to process: ").strip()
            if not file_path_input:
                print("File path cannot be empty. Please try again.")
                continue
            
            if file_path_input.lower() in ['exit', 'quit']:
                self.logger.info("Exiting interactive session during file input.")
                return

            try:
                indexed_count, failed_count = self.process_and_index_files(file_path_input)
                print(f"Processed files: {indexed_count} documents indexed, {failed_count} files failed.")
                if indexed_count == 0 and failed_count > 0:
                    print("No documents were indexed. Please check the file path and file contents.")
                    continue
                break
            except FileNotFoundError as e:
                print(f"Error: {e}. Please enter a valid path.")
            except ValueError as e:
                 print(f"Error: {e}. Please check the input path or file type.")
            except Exception as e:
                print(f"An unexpected error occurred during file processing: {e}")
                self.logger.error(f"Unexpected error in interactive file processing: {e}")

        # Query loop
        while True:
            query = input("\nEnter your query: ").strip()
            if not query:
                print("Query cannot be empty. Please try again or type 'exit'/'quit'.")
                continue

            if query.lower() in ['exit', 'quit']:
                self.logger.info("Exiting interactive session.")
                break

            result = self.query_knowledge_base(query)
            
            current_stats = self.vector_store.get_index_stats()
            docs_in_store = current_stats.get('total_vector_count', 0)
            
            self.output_manager.save_and_display_results(result, (docs_in_store, 0))
            self.output_manager.save_json_results(result, (docs_in_store, 0))

        print("Thank you for using the Interactive RAG System!")

    def get_system_health(self) -> Dict:
        """Get comprehensive system health information"""
        try:
            # Get index stats
            index_stats = self.vector_store.get_index_stats()
            
            # Get component info
            embedding_info = self.embedding_manager.get_embedding_info()
            query_info = self.query_engine.get_model_info()
            
            health = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'index_stats': index_stats,
                'embedding_config': embedding_info,
                'query_config': query_info,
                'system_config': {
                    'chunk_size': self.config.chunk_size,
                    'similarity_threshold': self.config.similarity_threshold,
                    'max_context_docs': self.config.max_context_docs
                }
            }
            
            self.logger.info("System health check passed")
            return health
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_index(self) -> bool:
        """Clear all documents from the index (use with caution)"""
        self.logger.warning("Clearing all documents from index")
        return self.vector_store.delete_all_vectors()
