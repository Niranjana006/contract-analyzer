"""
Text Processing Utilities
========================
Functions for cleaning, preprocessing, and analyzing contract text
"""

import re
import nltk
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    pass  # punkt_tab might not be available in all NLTK versions

def clean_contract_text(text: str) -> str:
    """
    Clean and preprocess contract text
    """
    if not text:
        return ""
    
    # Remove extra whitespaces and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,;:()$%/-]', '', text)
    
    # Fix common OCR issues
    text = text.replace('1l', 'II')  # Common OCR error
    text = text.replace('0', 'O')    # Sometimes O and 0 get confused
    
    return text.strip()

def split_into_clauses(text: str) -> List[str]:
    """
    Split contract text into individual clauses/sections
    """
    if not text:
        return []
    
    # Split by common section indicators
    section_patterns = [
        r'\n\d+\.',  # Numbered sections (1., 2., etc.)
        r'\n[A-Z]\.',  # Lettered sections (A., B., etc.)
        r'\n\([a-z]\)',  # Subsections (a), (b), etc.
        r'\n[A-Z][A-Z\s]+:',  # ALL CAPS headers
        r'\n\n'  # Double line breaks
    ]
    
    # Try to split using section patterns
    clauses = []
    current_text = text
    
    for pattern in section_patterns:
        splits = re.split(pattern, current_text)
        if len(splits) > 1:
            clauses.extend([clause.strip() for clause in splits if clause.strip()])
            break
    
    # If no clear sections found, split by paragraphs
    if not clauses:
        paragraphs = text.split('\n\n')
        clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    # If still no good splits, split by sentences for very long text
    if not clauses and len(text) > 1000:
        try:
            sentences = nltk.sent_tokenize(text)
            # Group sentences into logical chunks
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 500:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        clauses.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk:
                clauses.append(current_chunk.strip())
        except:
            # Fallback: split by periods
            clauses = [clause.strip() for clause in text.split('.') if len(clause.strip()) > 50]
    
    # Final fallback: return entire text as one clause
    if not clauses:
        clauses = [text]
    
    return clauses

def extract_key_info(text: str) -> Dict[str, List[str]]:
    """
    Extract key information from contract text using regex patterns
    """
    key_info = {
        'financial': [],
        'dates': [],
        'parties': [],
        'amounts': [],
        'deadlines': []
    }
    
    if not text:
        return key_info
    
    # Financial terms patterns
    money_patterns = [
        r'\$[\d,]+(?:\.\d{2})?',  # Dollar amounts
        r'[\d,]+\s*(?:dollars?|USD)',  # Written dollar amounts
        r'fee of \$?[\d,]+',  # Fee amounts
        r'payment of \$?[\d,]+',  # Payment amounts
    ]
    
    for pattern in money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['financial'].extend(matches)
    
    # Date patterns
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{1,2}\s+days?\b',
        r'\b\d{1,2}\s+(?:months?|years?)\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['dates'].extend(matches)
    
    # Party identification (basic)
    party_patterns = [
        r'between\s+([^,]+)\s+and\s+([^,\n]+)',
        r'"([^"]+)"\s*\(.*?Company.*?\)',
        r'"([^"]+)"\s*\(.*?Client.*?\)',
        r'Party\s*:\s*([^\n]+)',
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                key_info['parties'].extend(match)
            else:
                key_info['parties'].append(match)
    
    # Notice periods and deadlines
    deadline_patterns = [
        r'\b(\d+)\s+days?\s+(?:notice|prior|advance)',
        r'within\s+(\d+)\s+days?',
        r'not less than\s+(\d+)\s+days?',
        r'at least\s+(\d+)\s+days?'
    ]
    
    for pattern in deadline_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['deadlines'].extend([f"{match} days" for match in matches])
    
    # Clean and deduplicate
    for key in key_info:
        key_info[key] = list(set([item.strip() for item in key_info[key] if item.strip()]))
        key_info[key] = key_info[key][:5]  # Limit to top 5 items
    
    return key_info

def highlight_text(text: str, terms: List[str]) -> str:
    """
    Highlight specific terms in text for display
    """
    highlighted_text = text
    
    for term in terms:
        if term in highlighted_text:
            highlighted_text = highlighted_text.replace(
                term, 
                f"<mark style='background-color: yellow;'>{term}</mark>"
            )
    
    return highlighted_text

def extract_contract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata about the contract
    """
    metadata = {
        'word_count': len(text.split()),
        'character_count': len(text),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        'estimated_read_time': len(text.split()) // 200,  # Assuming 200 WPM
        'complexity_score': calculate_complexity_score(text)
    }
    
    return metadata

def calculate_complexity_score(text: str) -> float:
    """
    Calculate a simple complexity score based on sentence length and vocabulary
    """
    if not text:
        return 0.0
    
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0
            
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Simple complexity based on average sentence length
        # Scale from 0-10 where 10 is most complex
        complexity = min(10.0, avg_sentence_length / 3.0)
        
        return complexity
    except:
        return 5.0  # Default moderate complexity

def get_text_statistics(text: str) -> Dict[str, int]:
    """
    Get comprehensive text statistics
    """
    stats = {
        'characters': len(text),
        'words': len(text.split()),
        'sentences': len(re.findall(r'[.!?]+', text)),
        'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
        'unique_words': len(set(text.lower().split()))
    }
    
    return stats