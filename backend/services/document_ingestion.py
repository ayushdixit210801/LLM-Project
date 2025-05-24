import fitz
import uuid
import os
from typing import List, Dict, Tuple
from .vector_store import add_documents
from sqlalchemy.orm import Session
from ..models.database import Document as DBDocument

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately max_chunk_size characters"""
    # Split into sentences
    sentences = text.replace('\n', ' ').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size:
            if current_chunk:  # Only append if there's content
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def process_pdf(file_path: str, original_filename: str, db: Session) -> Tuple[str, DBDocument]:
    """
    Process a PDF file:
    1. Extract text from PDF
    2. Split into chunks
    3. Add to vector store
    4. Store metadata in database
    """
    # Generate unique ID for the document
    doc_id = str(uuid.uuid4())
    
    try:
        # Open the PDF
        doc = fitz.open(file_path)
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Split text into chunks
            chunks = chunk_text(text)
            
            # Create metadata and IDs for each chunk
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            chunk_metadatas = [{"page_number": page_num + 1, "document_id": doc_id} for _ in chunks]
            
            # Add to collections
            all_chunks.extend(chunks)
            all_metadatas.extend(chunk_metadatas)
            all_ids.extend(chunk_ids)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create database entry
        db_document = DBDocument(
            id=doc_id,
            filename=original_filename,
            total_pages=len(doc),
            total_chunks=len(all_chunks),
            file_size=file_size,
            status="processing"
        )
        
        # Add to database
        db.add(db_document)
        db.commit()
        
        # Close the document
        doc.close()
        
        # Add to vector store if we have chunks
        if all_chunks:
            await add_documents(all_chunks, all_metadatas, all_ids)
            
            # Update status to processed
            db_document.status = "processed"
            db.commit()
        
        return "success", db_document
        
    except Exception as e:
        # Update status to failed if document was created
        if 'db_document' in locals():
            db_document.status = "failed"
            db.commit()
        raise e