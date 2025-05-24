from google import genai
from google.genai import types
from typing import List, Tuple, Dict
import os
from .vector_store import search_documents as vector_search

# Configure Genai client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def format_prompt(query: str, context_chunks: List[Tuple[str, dict]]) -> str:
    """
    Format the prompt for Gemini with query and context
    """
    context_text = "\n\n".join([
        f"[Source: Page {meta['page_number']}]\n{text}"
        for text, meta in context_chunks
    ])
    
    prompt = f"""Based on the following context from a PDF document, please answer the question. 
If you cannot answer the question based on the provided context, please say so.

Context:
{context_text}

Question: {query}

Answer:"""
    
    return prompt

async def query_documents(query: str, n_results: int = 3) -> List[Tuple[str, Dict]]:
    """
    Query the document store and return relevant chunks with their metadata
    """
    # Search for relevant documents
    results = await vector_search(query, n_results)
    return results

async def query_documents_old(query: str) -> Tuple[str, List[str]]:
    """
    Process a query through the RAG pipeline:
    1. Retrieve relevant chunks from vector store
    2. Format prompt with context
    3. Generate answer using Gemini
    Returns tuple of (answer, source_list)
    """
    # Get relevant chunks
    context_chunks = await vector_search(query, n_results=3)
    
    if not context_chunks:
        return "I couldn't find any relevant information in the document to answer your question.", []
    
    # Format prompt
    prompt = format_prompt(query, context_chunks)
    
    try:
        # Generate response using the models API with configuration
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=1000,  # Allow longer responses for summarization
                temperature=0.3  # Lower temperature for more focused responses
            )
        )
        
        # Extract source pages
        sources = [f"Page {chunk[1]['page_number']}" for chunk in context_chunks]
        
        return response.text, list(set(sources))  # Deduplicate sources
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return "Sorry, I encountered an error while processing your request.", [] 