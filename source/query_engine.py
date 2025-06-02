"""
Query processing and response generation module for the RAG system.
"""

import logging
import time # Added import for time.sleep
from typing import List, Dict
import openai
import os # Added import for os.getenv

from .config import RAGConfig


class QueryEngine:
    """Handle query processing and response generation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger('rag_system.query_engine')
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client for chat completions"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.logger.info("Query engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize query engine: {e}")
            raise
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using context documents"""
        try:
            if not context_docs:
                return "I couldn't find any relevant information in the knowledge base to answer your question."
            
            # Prepare context with content validation
            context_parts = []
            valid_docs = 0
            
            for i, doc in enumerate(context_docs):
                text_content = doc['text']
                
                # Skip documents with no text content
                if not text_content.strip():
                    self.logger.warning(f"Skipping document {doc['id']} - no text content")
                    continue
                
                valid_docs += 1
                metadata = doc['metadata']
                source_info = f"[Source: {metadata.get('file_name', 'Unknown')}]"
                
                context_part = f"""Document {valid_docs} {source_info} (Relevance: {doc['score']:.3f}):
{text_content}"""
                context_parts.append(context_part)
            
            if not context_parts:
                return "I found potentially relevant documents, but they appear to contain no readable text content. Please check if the source files are properly formatted."
            
            context = "\n\n".join(context_parts)
            
            # Create prompts
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response
            response = self._generate_with_retry(system_prompt, user_prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the chat model"""
        return """You are an expert assistant that provides comprehensive, accurate answers based on the provided context documents. 

Instructions:
1. Use ONLY information from the provided context documents
2. Provide detailed, well-structured answers
3. When referencing information, cite the document number and source file
4. If the context doesn't contain enough information, clearly state this limitation
5. Synthesize information from multiple sources when relevant
6. Maintain a professional, informative tone
7. If you find contradictory information, acknowledge and explain the discrepancies"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context"""
        return f"""Context Documents:

{context}

Query: {query}

Please provide a comprehensive answer based on the context documents above. Remember to cite specific documents and source files when referencing information."""
    
    def _generate_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
    
    def validate_query(self, query: str) -> bool:
        """Validate query input"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 10000:  # Reasonable limit
            raise ValueError("Query too long (max 10000 characters)")
        
        return True
    
    def get_model_info(self) -> Dict:
        """Get information about the chat model configuration"""
        return {
            'model': self.config.chat_model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'max_retries': self.config.max_retries
        }