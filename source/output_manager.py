"""
Output management module for the RAG system.
Handles formatting and saving results.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List # Added List for create_summary_report
import json # Added import for save_json_results

from .utils import ensure_directory, format_timestamp


class OutputManager:
    """Handle output formatting and file writing"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = ensure_directory(Path(output_dir))
        self.logger = logging.getLogger('rag_system.output_manager')
    
    def save_and_display_results(self, result: Dict, file_stats: Tuple[int, int]) -> str:
        """Save results to file and display in terminal"""
        # Generate timestamp filename
        timestamp = format_timestamp()
        output_file = self.output_dir / f"{timestamp}.txt"
        
        # Format output
        output_content = self._format_output(result, file_stats)
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            self.logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results to file: {e}")
        
        # Display in terminal
        self._display_results(output_content, output_file)
        
        return str(output_file)
    
    def _format_output(self, result: Dict, file_stats: Tuple[int, int]) -> str:
        """Format output content"""
        indexed_docs, failed_files = file_stats
        
        output_lines = [
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"FILE PROCESSING SUMMARY:",
            f"- Documents successfully indexed: {indexed_docs}",
            f"- Files that failed processing: {failed_files}",
            f"",
            f"QUERY: {result['query']}",
            f"",
            f"ANSWER:",
            f"{result['answer']}",
            f"",
            f"RETRIEVAL STATISTICS:",
            f"- Total matches found: {result.get('total_matches', 0)}",
            f"- Relevant matches used: {result.get('relevant_matches', 0)}",
            f""
        ]
        
        if result.get('sources'):
            output_lines.append("SOURCE DOCUMENTS:")
            for i, source in enumerate(result['sources'], 1):
                metadata = source['metadata']
                file_name = metadata.get('file_name', 'Unknown')
                chunk_info = f"chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                
                # Show actual preview of text content
                preview_text = source['text'][:150] if source['text'] else "No text content"
                
                output_lines.extend([
                    f"{i}. {file_name} ({chunk_info}) - Relevance: {source['score']:.3f}",
                    f"   Path: {metadata.get('source_file', 'Unknown')}",
                    f"   Type: {metadata.get('file_type', 'Unknown')}",
                    f"   Preview: {preview_text}{'...' if len(preview_text) > 150 else ''}", # Corrected from len(preview_text) > 150 to len(source['text']) > 150
                    ""
                ])
        else:
            output_lines.append("No relevant source documents found.")
        
        if 'error' in result:
            output_lines.extend([
                "",
                f"ERROR: {result['error']}"
            ])
        
        return "\n".join(output_lines)
    
    def _display_results(self, content: str, output_file: Path):
        """Display results in terminal"""
        print("\n" + "="*80)
        print("RETRIEVAL AUGMENTED GENERATION RESULTS")
        print("="*80)
        print(content)
        print("="*80)
        print(f"\nResults saved to: {output_file}")
    
    def save_json_results(self, result: Dict, file_stats: Tuple[int, int]) -> str:
        """Save results in JSON format"""
        timestamp = format_timestamp()
        output_file = self.output_dir / f"{timestamp}.json"
        
        # Prepare JSON data
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'file_stats': {
                'indexed_documents': file_stats[0],
                'failed_files': file_stats[1]
            },
            'result': result
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"JSON results saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            return ""
    
    def create_summary_report(self, results: List[Dict]) -> str:
        """Create a summary report from multiple query results"""
        timestamp = format_timestamp()
        output_file = self.output_dir / f"summary_{timestamp}.txt"
        
        summary_lines = [
            f"RAG System Summary Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Queries: {len(results)}",
            f"",
            "="*80,
            ""
        ]
        
        for i, result in enumerate(results, 1):
            summary_lines.extend([
                f"QUERY {i}: {result['query']}",
                f"Relevant Sources: {result.get('relevant_matches', 0)}",
                f"Answer Preview: {result['answer'][:200]}...",
                "",
                "-"*40,
                ""
            ])
        
        content = "\n".join(summary_lines)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Summary report saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {e}")
            return ""
