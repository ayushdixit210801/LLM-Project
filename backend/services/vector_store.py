from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np
import faiss
import os
import json

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embedding dimension
EMBEDDING_DIM = model.get_sentence_embedding_dimension()

# Initialize FAISS index
index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product index (cosine similarity after normalization)

# Initialize storage for metadata
documents = []
metadatas = []

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity"""
    return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

def save_state():
    """Save the index and documents to disk"""
    if not os.path.exists('vector_db'):
        os.makedirs('vector_db')
    
    # Save FAISS index
    faiss.write_index(index, 'vector_db/vectors.faiss')
    
    # Save documents and metadata
    with open('vector_db/metadata.json', 'w') as f:
        json.dump({
            'documents': documents,
            'metadata': metadatas
        }, f)

def load_state():
    """Load the index and documents from disk"""
    global index, documents, metadatas
    
    if os.path.exists('vector_db/vectors.faiss') and os.path.exists('vector_db/metadata.json'):
        # Load FAISS index
        index = faiss.read_index('vector_db/vectors.faiss')
        
        # Load documents and metadata
        with open('vector_db/metadata.json', 'r') as f:
            data = json.load(f)
            documents = data['documents']
            metadatas = data['metadata']

async def add_documents(texts: List[str], metadatas_list: List[Dict], ids: List[str]) -> None:
    """
    Add documents to the vector store with their embeddings
    """
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Normalize vectors for cosine similarity
    embeddings = normalize_vectors(embeddings)
    
    # Add to FAISS index
    index.add(embeddings.astype(np.float32))
    
    # Add to storage
    documents.extend(texts)
    metadatas.extend(metadatas_list)
    
    # Save state
    save_state()

async def search_documents(query: str, n_results: int = 3, document_id: str = None) -> List[Tuple[str, Dict]]:
    """
    Search for relevant documents using the query
    Args:
        query: The search query
        n_results: Number of results to return
        document_id: Optional document ID to filter results
    Returns list of (text, metadata) tuples
    """
    if index.ntotal == 0:
        return []
    
    # Generate query embedding and normalize
    query_embedding = model.encode([query])[0]
    query_embedding = normalize_vectors(query_embedding.reshape(1, -1))
    
    # Search in FAISS index
    scores, indices = index.search(query_embedding.astype(np.float32), index.ntotal)
    
    # Get all results and filter by document_id if provided
    results = []
    seen_indices = set()
    
    for idx in indices[0]:
        # Skip if we've seen this index or score is 0
        if idx in seen_indices or scores[0][len(seen_indices)] <= 0:
            continue
            
        # Get document and metadata
        doc = documents[idx]
        meta = metadatas[idx]
        
        # Filter by document_id if provided
        if document_id and meta.get('document_id') != document_id:
            continue
            
        results.append((doc, meta))
        seen_indices.add(idx)
        
        # Break if we have enough results
        if len(results) >= n_results:
            break
    
    return results

# Try to load existing state on module import
try:
    load_state()
except:
    pass  # If loading fails, we'll start with empty storage 