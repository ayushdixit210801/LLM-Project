from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import os
import tempfile
from google import genai
from google.genai import types
from backend.services.document_ingestion import process_pdf
from backend.services.rag_pipeline import query_documents
from backend.models.database import get_db
from sqlalchemy.orm import Session
from backend.schemas.document import Document, DocumentList, DocumentStats
from sqlalchemy import func
from backend.models.database import Document as DBDocument

# Configure Genai client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

class QueryRequest(BaseModel):
    docId: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process the PDF
        result, document = await process_pdf(temp_path, file.filename, db)
        
        # Clean up
        os.unlink(temp_path)
        
        return {"message": "PDF processed successfully", "document": Document.from_orm(document)}
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query a specific document"""
    try:
        # Verify document exists
        document = db.query(DBDocument).filter(DBDocument.id == request.docId).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get relevant documents
        results = await query_documents(request.query, document_id=request.docId)
        
        if not results:
            return QueryResponse(
                answer="I don't have enough context to answer your question about this document.",
                sources=[]
            )
        
        # Format context for Gemini
        context = "\n\n".join([
            f"[Page {meta['page_number']}]\n{text}" 
            for text, meta in results
        ])
        
        # Generate prompt
        prompt = f"""Based on the following context from the specified PDF document, please answer the question.
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
                sources=[f"Page {meta['page_number']}" for _, meta in results]
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

@router.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get specific document metadata"""
    document = db.query(DBDocument).filter(DBDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document