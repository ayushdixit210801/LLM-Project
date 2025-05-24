from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import numpy as np
import pickle
import os

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize storage
embeddings = []
documents = []
metadatas = []

def save_state():
    """Save the index and documents to disk"""
    if not os.path.exists('vector_db'):
        os.makedirs('vector_db')
    
    with open('vector_db/store.pkl', 'wb') as f:
        pickle.dump((embeddings, documents, metadatas), f)

def load_state():
    """Load the index and documents from disk"""
    global embeddings, documents, metadatas
    
    if os.path.exists('vector_db/store.pkl'):
        with open('vector_db/store.pkl', 'rb') as f:
            embeddings, documents, metadatas = pickle.load(f)

async def add_documents(texts: List[str], metadatas_list: List[Dict], ids: List[str]) -> None:
    """
    Add documents to the vector store with their embeddings
    """
    # Generate embeddings
    new_embeddings = model.encode(texts)
    
    # Add to storage
    embeddings.extend(new_embeddings)
    documents.extend(texts)
    metadatas.extend(metadatas_list)
    
    # Save state
    save_state()

async def search_documents(query: str, n_results: int = 3) -> List[Tuple[str, Dict]]:
    """
    Search for relevant documents using the query
    Returns list of (text, metadata) tuples
    """
    if not embeddings:
        return []
    
    # Generate query embedding
    query_embedding = model.encode([query])[0].reshape(1, -1)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    
    # Return results
    results = []
    for idx in top_indices:
        results.append((documents[idx], metadatas[idx]))
    
    return results

# Try to load existing state on module import
try:
    load_state()
except:
    pass  # If loading fails, we'll start with empty storage 