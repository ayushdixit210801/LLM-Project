from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocumentBase(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    file_size: int
    status: str

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: str
    upload_date: datetime

    class Config:
        from_attributes = True

class DocumentList(BaseModel):
    total: int
    documents: List[Document]

class DocumentStats(BaseModel):
    total_documents: int
    total_pages: int
    total_chunks: int
    total_size: int  # in bytes