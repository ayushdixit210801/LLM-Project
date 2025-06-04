from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import os
import tempfile
from google import genai
from google.genai import types
from backend.services.document_ingestion import process_pdf
from backend.services.rag_pipeline import query_documents
from backend.models.database import get_db, engine, Base
from sqlalchemy.orm import Session
from backend.schemas.document import Document, DocumentList, DocumentStats
from sqlalchemy import func
from backend.models.database import Document as DBDocument
import fitz  # PyMuPDF
import shutil
from pathlib import Path

# Configure Genai client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadResponse(BaseModel):
    message: str
    documents: List[Document]

def validate_pdf_pages(file_path: str, max_pages: int = 1000) -> bool:
    """Validate that PDF has no more than max_pages"""
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        return page_count <= max_pages
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error validating PDF: {str(e)}")

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process multiple PDF files (up to 20)"""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 PDF files allowed per upload")
    
    if not all(file.filename.endswith('.pdf') for file in files):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    processed_documents = []
    temp_files = []
    
    try:
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            # Validate page count before processing
            if not validate_pdf_pages(temp_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF file '{file.filename}' exceeds maximum page limit of 1000 pages"
                )
            
            # Process the PDF
            result, document = await process_pdf(temp_path, file.filename, db)
            processed_documents.append(Document.from_orm(document))
        
        return UploadResponse(
            message=f"Successfully processed {len(processed_documents)} PDF files",
            documents=processed_documents
        )
    
    except HTTPException as he:
        # Clean up on validation error
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise he
    
    except Exception as e:
        # Clean up on processing error
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query across all documents"""
    try:
        # Get relevant documents from all available documents
        results = await query_documents(request.query)
        
        if not results:
            return QueryResponse(
                answer="I don't have enough context to answer your question from the available documents.",
                sources=[]
            )
        
        # Format context for Gemini
        context = "\n\n".join([
            f"[Document: {meta['document_id']}, Page {meta['page_number']}]\n{text}" 
            for text, meta in results
        ])
        
        # Generate prompt
        prompt = f"""Based on the following context from the available PDF documents, please answer the question.
If you cannot answer based on the provided context, please say so.

Context:
{context}

Question: {request.query}

Answer:"""
        
        try:
            # Generate response using the models API
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0.3
                )
            )
            
            # Return response with sources
            return QueryResponse(
                answer=response.text,
                sources=[f"Document: {meta['document_id']}, Page {meta['page_number']}" for _, meta in results]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List all processed documents with pagination"""
    total = db.query(DBDocument).count()
    documents = db.query(DBDocument).offset(skip).limit(limit).all()
    return DocumentList(total=total, documents=documents)