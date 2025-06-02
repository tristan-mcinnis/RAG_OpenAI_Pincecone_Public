#!/usr/bin/env python3
"""
Main entry point for the RAG system.
Provides CLI for processing files and querying.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "source"))

from source.rag_system import ProductionRAGSystem
from source.config import RAGConfig
from source.utils import validate_environment, setup_logging, format_timestamp


def main():
    """Main function to run the RAG system CLI"""
    
    # Load environment variables
    load_dotenv()
    
    # Setup argument parser first
    parser = argparse.ArgumentParser(
        description="Production RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process_query "/path/to/file.txt" "What is this about?"
  python main.py process "/path/to/documents/"
  python main.py query "What did we learn about the topic?"
  python main.py retrieve "search terms" --top-k 15 --min-score 0.3
  python main.py extract_verbatims "basketball opinions" --min-length 30 --exclude-moderator
  python main.py interactive
  python main.py health_check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process files command
    process_parser = subparsers.add_parser('process', help='Process and index files')
    process_parser.add_argument('path', type=str, help='Path to file or directory to process')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('query_text', type=str, help='Query string')

    # Process and query command (most common use case)
    process_query_parser = subparsers.add_parser('process_query', help='Process files and then query')
    process_query_parser.add_argument('path', type=str, help='Path to file or directory to process')
    process_query_parser.add_argument('query_text', type=str, help='Query string')

    # Retrieve only command (no response generation)
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve relevant documents without generating response')
    retrieve_parser.add_argument('query_text', type=str, help='Search query')
    retrieve_parser.add_argument('--top-k', type=int, default=10, help='Number of documents to retrieve (default: 10)')
    retrieve_parser.add_argument('--min-score', type=float, default=0.1, help='Minimum similarity score (default: 0.1)')

    # Extract verbatims command (NEW)
    verbatim_parser = subparsers.add_parser('extract_verbatims', help='Extract clean verbatims with speaker attribution')
    verbatim_parser.add_argument('query_text', type=str, help='Search query for verbatim extraction')
    verbatim_parser.add_argument('--min-length', type=int, default=20, help='Minimum quote length in characters (default: 20)')
    verbatim_parser.add_argument('--max-length', type=int, default=500, help='Maximum quote length in characters (default: 500)')
    verbatim_parser.add_argument('--exclude-moderator', action='store_true', default=True, help='Exclude moderator quotes (default: True)')
    verbatim_parser.add_argument('--include-moderator', action='store_true', help='Include moderator quotes')
    verbatim_parser.add_argument('--participant-filter', type=str, help='Filter by participant criteria (e.g., "M, 18-24")')
    verbatim_parser.add_argument('--format', choices=['research', 'quotes_only', 'detailed', 'csv'], default='research', help='Output format (default: research)')
    verbatim_parser.add_argument('--top-k', type=int, default=20, help='Number of documents to search (default: 20)')
    verbatim_parser.add_argument('--export-csv', type=str, help='Export to CSV file (provide filename)')

    # Interactive mode command
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')

    # Health check command
    health_parser = subparsers.add_parser('health_check', help='Check system health')

    # Delete all vectors command (use with caution)
    delete_parser = subparsers.add_parser('delete_all', help='Delete all vectors from the index (DANGEROUS!)')

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return

    # Setup logging (early setup for any issues during init)
    logger = setup_logging()
    
    try:
        # Validate environment variables
        validate_environment()
        logger.info("Environment validation passed")
        
    except EnvironmentError as e:
        logger.error(f"Environment validation failed: {e}")
        print(f"❌ Environment Error: {e}")
        print("\n💡 Please check your .env file and ensure you have:")
        print("   • OPENAI_API_KEY=your_actual_openai_key")
        print("   • PINECONE_API_KEY=your_actual_pinecone_key")
        print("   • Make sure these are real API keys, not placeholder text!")
        return

    # Handle health check without full initialization
    if args.command == 'health_check':
        try:
            config = RAGConfig.from_env()
            rag_system = ProductionRAGSystem(config)
            health = rag_system.get_system_health()
            
            print(f"\n🔍 System Health Check")
            print(f"Status: {health['status'].upper()}")
            
            if health['status'] == 'healthy':
                print("✅ All systems operational")
                if 'index_stats' in health:
                    stats = health['index_stats']
                    print(f"📊 Index Stats: {stats.get('total_vector_count', 0)} vectors")
            else:
                print(f"❌ System unhealthy: {health.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            print(f"❌ Health check failed: {e}")
        return

    # Initialize RAG system for other commands
    try:
        config = RAGConfig.from_env()
        print("🚀 Initializing RAG System...")
        rag_system = ProductionRAGSystem(config)
        print("✅ RAG System initialized successfully")
        
    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        print(f"❌ Initialization failed: {e}")
        
        # Provide helpful error context
        if "401" in str(e) or "invalid_api_key" in str(e):
            print("\n💡 This looks like an API key issue. Please:")
            print("   1. Check your .env file")
            print("   2. Ensure OPENAI_API_KEY contains your real OpenAI API key")
            print("   3. Ensure PINECONE_API_KEY contains your real Pinecone API key")
            print("   4. Remove any quotes around the keys in .env")
        
        return

    # Execute commands
    try:
        if args.command == 'process':
            logger.info(f"Processing files from: {args.path}")
            print(f"\n📁 Processing files from: {args.path}")
            
            indexed_docs, failed_files = rag_system.process_and_index_files(args.path)
            print(f"✅ File processing complete!")
            print(f"   • Indexed documents: {indexed_docs}")
            print(f"   • Failed files: {failed_files}")
        
        elif args.command == 'query':
            logger.info(f"Querying with: {args.query_text[:50]}...")
            print(f"\n🔍 Querying knowledge base...")
            
            result = rag_system.query_knowledge_base(args.query_text)
            
            # Get current stats for display
            current_stats = rag_system.vector_store.get_index_stats()
            docs_in_store = current_stats.get('total_vector_count', 0)
            
            output_file = rag_system.output_manager.save_and_display_results(result, (docs_in_store, 0))
            print(f"\n📄 Results saved to: {output_file}")
        
        elif args.command == 'retrieve':
            logger.info(f"Retrieving documents for: {args.query_text[:50]}...")
            print(f"\n🔍 Retrieving relevant documents...")
            print(f"📝 Query: {args.query_text}")
            print(f"🎯 Top-K: {args.top_k}")
            print(f"📊 Min Score: {args.min_score}")
            
            # Create query embedding
            query_embedding = rag_system.embedding_manager.create_single_embedding(args.query_text)
            
            # Search for relevant documents
            relevant_docs = rag_system.vector_store.search(query_embedding, top_k=args.top_k)
            
            # Filter by minimum score if specified
            if args.min_score > 0.1:
                relevant_docs = [doc for doc in relevant_docs if doc['score'] >= args.min_score]
            
            # Display structured results
            print(f"\n📊 RETRIEVAL RESULTS")
            print("=" * 80)
            print(f"Found: {len(relevant_docs)} relevant documents")
            print("=" * 80)
            
            if not relevant_docs:
                print("❌ No documents found matching your criteria")
                print("💡 Try:")
                print("   • Lowering --min-score")
                print("   • Increasing --top-k")
                print("   • Using different search terms")
            else:
                for i, doc in enumerate(relevant_docs, 1):
                    metadata = doc['metadata']
                    print(f"\n📄 DOCUMENT {i}")
                    print(f"   🎯 Relevance Score: {doc['score']:.3f}")
                    print(f"   🆔 Document ID: {doc['id']}")
                    print(f"   📁 Source File: {metadata.get('file_name', 'Unknown')}")
                    print(f"   🧩 Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
                    print(f"   📍 Full Path: {metadata.get('source_file', 'Unknown')}")
                    print(f"   📝 Content Preview:")
                    print(f"      {doc['text'][:300]}...")
                    if len(doc['text']) > 300:
                        print(f"   📏 Total Length: {len(doc['text'])} characters")
                    print("-" * 80)
            
            # Save structured results to JSON
            timestamp = format_timestamp()
            output_file = rag_system.output_manager.output_dir / f"retrieval_{timestamp}.json"
            
            retrieval_data = {
                'query': args.query_text,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'top_k': args.top_k,
                    'min_score': args.min_score
                },
                'results': {
                    'total_found': len(relevant_docs),
                    'documents': relevant_docs
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Structured results saved to: {output_file}")
        
        elif args.command == 'extract_verbatims':
            logger.info(f"Extracting verbatims for: {args.query_text[:50]}...")
            print(f"\n💬 Extracting verbatims...")
            print(f"📝 Query: {args.query_text}")
            print(f"📏 Length: {args.min_length}-{args.max_length} characters")
            print(f"🎭 Exclude Moderator: {not args.include_moderator}")
            print(f"🎯 Format: {args.format}")
            
            # Handle moderator inclusion/exclusion
            exclude_moderator = not args.include_moderator
            
            # Extract verbatims
            result = rag_system.extract_verbatims(
                query=args.query_text,
                min_length=args.min_length,
                max_length=args.max_length,
                exclude_moderator=exclude_moderator,
                participant_filter=args.participant_filter,
                format_style=args.format,
                top_k=args.top_k
            )
            
            # Display results
            print(f"\n💬 VERBATIM EXTRACTION RESULTS")
            print("=" * 80)
            print(f"Found: {result['total_found']} verbatims")
            print("=" * 80)
            
            if result['total_found'] == 0:
                print("❌ No verbatims found matching your criteria")
                print("💡 Try:")
                print("   • Lowering --min-length")
                print("   • Increasing --max-length")
                print("   • Including moderator with --include-moderator")
                print("   • Using different search terms")
            else:
                if args.format == 'csv':
                    print("\n📄 CSV FORMAT:")
                    print(result['formatted_verbatims'])
                else:
                    print(f"\n📄 FORMATTED VERBATIMS ({args.format.upper()}):")
                    for i, verbatim in enumerate(result['formatted_verbatims'], 1):
                        print(f"\n{i}. {verbatim}")
                
                # Show summary
                print(f"\n📊 SUMMARY:")
                speakers = set(v['speaker'] for v in result['verbatims'])
                avg_words = sum(v['word_count'] for v in result['verbatims']) / len(result['verbatims'])
                print(f"   • Unique speakers: {len(speakers)}")
                print(f"   • Average words per quote: {avg_words:.1f}")
                print(f"   • Speaker breakdown: {', '.join(speakers)}")
            
            # Save results
            timestamp = format_timestamp()
            
            # Save JSON data
            output_file = rag_system.output_manager.output_dir / f"verbatims_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Export CSV if requested
            if args.export_csv:
                csv_content = rag_system.verbatim_extractor.export_to_csv(
                    [rag_system.verbatim_extractor.verbatim_extractor.Verbatim(**v) for v in result['verbatims']]
                )
                with open(args.export_csv, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
                print(f"📊 CSV exported to: {args.export_csv}")
        
        elif args.command == 'process_query':
            logger.info(f"Processing files and querying: {args.query_text[:50]}...")
            print(f"\n🔄 Processing files and querying...")
            print(f"📁 Path: {args.path}")
            print(f"❓ Query: {args.query_text}")
            
            result, file_stats = rag_system.process_and_query(args.path, args.query_text)
            
            if file_stats[0] == 0:
                print("❌ No documents were processed successfully")
                return
                
            output_file = rag_system.output_manager.save_and_display_results(result, file_stats)
            print(f"\n📄 Results saved to: {output_file}")
            
        elif args.command == 'interactive':
            print("🎮 Starting interactive mode...")
            rag_system.run_interactive_session()

        elif args.command == 'delete_all':
            print("⚠️  WARNING: This will delete ALL vectors from the index!")
            confirm = input("Are you sure? Type 'DELETE_ALL' to confirm: ")
            if confirm == 'DELETE_ALL':
                logger.warning("User initiated deletion of all vectors.")
                if rag_system.vector_store.delete_all_vectors():
                    print("✅ All vectors deleted from index")
                else:
                    print("❌ Failed to delete vectors")
            else:
                print("Operation cancelled")
                
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ File not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
