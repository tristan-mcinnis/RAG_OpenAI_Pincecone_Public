"""
Verbatim extraction module for the RAG system.
Extracts clean, attributed quotes from transcript data.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import RAGConfig


class VerbatimFormat(Enum):
    """Output format options for verbatims"""
    RESEARCH = "research"
    QUOTES_ONLY = "quotes_only"
    DETAILED = "detailed"
    CSV = "csv"


@dataclass
class SpeakerInfo:
    """Container for speaker information"""
    name: str
    demographics: str
    location: str
    raw_identifier: str


@dataclass
class Verbatim:
    """Container for extracted verbatim"""
    quote: str
    speaker: SpeakerInfo
    relevance_score: float
    source_chunk: str
    timestamp: str
    cleaned_quote: str
    word_count: int


class VerbatimExtractor:
    """Extract and format verbatims from transcript data"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger('rag_system.verbatim_extractor')
        
        # Regex patterns for parsing transcript format
        self.speaker_pattern = re.compile(
            r'([^,\[]+(?:,\s*[^,\[]+)*),?\s*\[([^\]]+)\]:\s*(.*?)(?=\n[A-Z]|\n$|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        # Pattern for extracting demographics from speaker info
        self.demographics_pattern = re.compile(
            r'([^,]+),\s*([MF]),\s*(\d+-\d+),\s*([A-Z]+)'
        )
        
        # Moderator patterns
        self.moderator_patterns = ['Moderator', 'MODERATOR', 'Mod', 'MOD']
    
    def extract_verbatims(
        self,
        query: str,
        retrieved_docs: List[Dict],
        min_length: int = 20,
        max_length: int = 500,
        exclude_moderator: bool = True,
        participant_filter: Optional[str] = None,
        format_style: VerbatimFormat = VerbatimFormat.RESEARCH,
        include_context: bool = False
    ) -> List[Verbatim]:
        """
        Extract verbatims from retrieved documents
        
        Args:
            query: Original search query
            retrieved_docs: Documents from vector search
            min_length: Minimum quote length in characters
            max_length: Maximum quote length in characters
            exclude_moderator: Whether to exclude moderator quotes
            participant_filter: Filter by participant criteria (e.g., "M, 18-24")
            format_style: Output format style
            include_context: Whether to include surrounding context
            
        Returns:
            List of extracted and formatted verbatims
        """
        
        self.logger.info(f"Extracting verbatims for query: '{query[:50]}...'")
        
        all_verbatims = []
        
        for doc in retrieved_docs:
            verbatims = self._extract_from_document(
                doc, min_length, max_length, exclude_moderator, 
                participant_filter, include_context
            )
            all_verbatims.extend(verbatims)
        
        # Sort by relevance score (highest first)
        all_verbatims.sort(key=lambda v: v.relevance_score, reverse=True)
        
        self.logger.info(f"Extracted {len(all_verbatims)} verbatims")
        return all_verbatims
    
    def _extract_from_document(
        self,
        doc: Dict,
        min_length: int,
        max_length: int,
        exclude_moderator: bool,
        participant_filter: Optional[str],
        include_context: bool
    ) -> List[Verbatim]:
        """Extract verbatims from a single document"""
        
        text = doc['text']
        relevance_score = doc['score']
        
        verbatims = []
        
        # Find all speaker segments in the text
        matches = self.speaker_pattern.findall(text)
        
        for match in matches:
            speaker_info_raw, timestamp, quote_content = match
            
            try:
                # Parse speaker information
                speaker_info = self._parse_speaker_info(speaker_info_raw.strip())
                
                if not speaker_info:
                    continue
                
                # Skip moderator if requested
                if exclude_moderator and self._is_moderator(speaker_info.name):
                    continue
                
                # Clean the quote
                cleaned_quote = self._clean_quote(quote_content.strip())
                
                # Apply length filters
                if len(cleaned_quote) < min_length or len(cleaned_quote) > max_length:
                    continue
                
                # Apply participant filter if specified
                if participant_filter and not self._matches_participant_filter(
                    speaker_info, participant_filter
                ):
                    continue
                
                # Create verbatim object
                verbatim = Verbatim(
                    quote=quote_content.strip(),
                    speaker=speaker_info,
                    relevance_score=relevance_score,
                    source_chunk=text,
                    timestamp=timestamp.strip(),
                    cleaned_quote=cleaned_quote,
                    word_count=len(cleaned_quote.split())
                )
                
                verbatims.append(verbatim)
                
            except Exception as e:
                self.logger.warning(f"Error processing speaker segment: {e}")
                continue
        
        return verbatims
    
    def _parse_speaker_info(self, speaker_info_raw: str) -> Optional[SpeakerInfo]:
        """Parse speaker information from transcript format"""
        
        # Try to match demographic pattern: Name, Gender, Age, Location
        demo_match = self.demographics_pattern.match(speaker_info_raw)
        
        if demo_match:
            name, gender, age, location = demo_match.groups()
            demographics = f"{gender}, {age}"
            
            return SpeakerInfo(
                name=name.strip(),
                demographics=demographics,
                location=location,
                raw_identifier=speaker_info_raw
            )
        
        # Handle moderator or simpler formats
        if any(mod in speaker_info_raw for mod in self.moderator_patterns):
            return SpeakerInfo(
                name="Moderator",
                demographics="",
                location="",
                raw_identifier=speaker_info_raw
            )
        
        # Handle other formats (just use as name)
        return SpeakerInfo(
            name=speaker_info_raw,
            demographics="",
            location="",
            raw_identifier=speaker_info_raw
        )
    
    def _clean_quote(self, quote: str) -> str:
        """Clean up quote text for better readability"""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', quote).strip()
        
        # Remove common speech artifacts
        cleaned = re.sub(r'\b(um|uh|like|you know)\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove trailing periods if quote seems incomplete
        cleaned = re.sub(r'\.$', '', cleaned)
        
        # Remove extra spaces after cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _is_moderator(self, speaker_name: str) -> bool:
        """Check if speaker is a moderator"""
        return any(mod.lower() in speaker_name.lower() for mod in self.moderator_patterns)
    
    def _matches_participant_filter(self, speaker_info: SpeakerInfo, filter_criteria: str) -> bool:
        """Check if speaker matches participant filter criteria"""
        
        filter_parts = [part.strip() for part in filter_criteria.split(',')]
        
        for part in filter_parts:
            if part.upper() in speaker_info.demographics.upper():
                return True
            if part.upper() in speaker_info.location.upper():
                return True
        
        return False
    
    def format_verbatims(
        self,
        verbatims: List[Verbatim],
        format_style: VerbatimFormat = VerbatimFormat.RESEARCH
    ) -> List[str]:
        """Format verbatims according to specified style"""
        
        formatted = []
        
        for verbatim in verbatims:
            if format_style == VerbatimFormat.RESEARCH:
                formatted_verbatim = self._format_research_style(verbatim)
            elif format_style == VerbatimFormat.QUOTES_ONLY:
                formatted_verbatim = self._format_quotes_only(verbatim)
            elif format_style == VerbatimFormat.DETAILED:
                formatted_verbatim = self._format_detailed_style(verbatim)
            else:
                formatted_verbatim = self._format_research_style(verbatim)
            
            formatted.append(formatted_verbatim)
        
        return formatted
    
    def _format_research_style(self, verbatim: Verbatim) -> str:
        """Format in research verbatim style"""
        
        # Build speaker attribution
        attribution_parts = [verbatim.speaker.name]
        
        if verbatim.speaker.location:
            attribution_parts.append(verbatim.speaker.location)
        
        if verbatim.speaker.demographics:
            attribution_parts.append(verbatim.speaker.demographics)
        
        attribution = ", ".join(attribution_parts)
        
        return f'"{verbatim.cleaned_quote}" - {attribution}'
    
    def _format_quotes_only(self, verbatim: Verbatim) -> str:
        """Format as quote only"""
        return f'"{verbatim.cleaned_quote}"'
    
    def _format_detailed_style(self, verbatim: Verbatim) -> str:
        """Format with detailed information"""
        
        attribution_parts = [verbatim.speaker.name]
        
        if verbatim.speaker.location:
            attribution_parts.append(verbatim.speaker.location)
        
        if verbatim.speaker.demographics:
            attribution_parts.append(verbatim.speaker.demographics)
        
        attribution = ", ".join(attribution_parts)
        
        return (f'"{verbatim.cleaned_quote}" - {attribution}\n'
                f'[Relevance: {verbatim.relevance_score:.3f} | '
                f'Words: {verbatim.word_count} | '
                f'Time: {verbatim.timestamp}]')
    
    def export_to_csv(self, verbatims: List[Verbatim]) -> str:
        """Export verbatims to CSV format"""
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Quote', 'Speaker', 'Demographics', 'Location', 
            'Relevance_Score', 'Word_Count', 'Timestamp'
        ])
        
        # Write verbatims
        for verbatim in verbatims:
            writer.writerow([
                verbatim.cleaned_quote,
                verbatim.speaker.name,
                verbatim.speaker.demographics,
                verbatim.speaker.location,
                f"{verbatim.relevance_score:.3f}",
                verbatim.word_count,
                verbatim.timestamp
            ])
        
        return output.getvalue()
