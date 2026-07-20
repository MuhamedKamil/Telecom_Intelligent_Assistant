# utils.py or language_utils.py
import re
from typing import Tuple

def detect_language(text: str) -> str:
    """
    Detect if text is primarily Arabic or English.
    
    Returns:
        'ar' for Arabic, 'en' for English
    """
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    english = len(re.findall(r'[A-Za-z]', text))
    
    if arabic > english:
        return "ar"
    return "en"

def get_language_confidence(text: str) -> Tuple[str, float]:
    """
    Detect language with confidence score.
    
    Returns:
        Tuple of (language, confidence) where confidence is 0-1
    """
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    english = len(re.findall(r'[A-Za-z]', text))
    total = max(arabic + english, 1)
    
    if arabic > english:
        return "ar", arabic / total
    return "en", english / total

