from typing import List
import re

def chunk_text(text: str, target_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of approximately target_size tokens with overlap.
    Uses sentence boundaries where possible to create more natural chunks.
    """
    # Clean and normalize text
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Rough token count (words + punctuation)
        sentence_size = len(sentence.split())
        
        # If adding this sentence would exceed target size, save current chunk
        if current_size + sentence_size > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Keep last few sentences for overlap
            overlap_tokens = 0
            overlap_chunk = []
            
            for s in reversed(current_chunk):
                overlap_tokens += len(s.split())
                if overlap_tokens >= overlap:
                    break
                overlap_chunk.insert(0, s)
            
            current_chunk = overlap_chunk
            current_size = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 