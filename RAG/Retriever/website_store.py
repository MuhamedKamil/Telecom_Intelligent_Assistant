import torch
from typing import List, Dict, Any, Optional

class WebsiteStore:
    """
    In-memory vector store for website content with semantic search capability.
    
    This class manages indexing, storing, and searching through website content
    by maintaining text chunks and their corresponding embeddings from crawled web pages.
    
    """
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]]       = []
        self.embeddings: Optional[torch.Tensor] = None

    def load(self, chunks: List[Dict[str, Any]], embeddings: torch.Tensor) -> None:
        """
        Load website chunks and their embeddings into the store.
        
        This replaces any existing data with the new chunks and embeddings.
        
        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with their metadata
                (e.g., url, title, page content snippet).
            embeddings (torch.Tensor): Tensor of shape (num_chunks, embedding_dim)
                containing dense vector representations for each chunk.
        """
        self.chunks     = chunks
        self.embeddings = embeddings

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:

        """
        Search for the most similar website chunks to a query embedding.
        
        Args:
            query_embedding (torch.Tensor): Query embedding vector of shape (embedding_dim,).
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of top-k relevant chunks, each enriched with:
                - All original chunk metadata (url, title, text, etc.)
                - 'score' (float): Similarity score (higher = more relevant)
                - 'source' (str): Always "website" to identify the source
            
        """
        if not self.chunks:
            return []

        scores = query_embedding @ self.embeddings.T
        values, indices = torch.topk(scores, k=min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            chunk = self.chunks[idx].copy()
            chunk["score"] = score
            chunk["source"] = "website"
            results.append(chunk)

        return results
        
    def clear(self):
        """
        Clear all stored website content from the store.
        
        This removes all chunks and embeddings to free memory.
        Useful when clearing cached website data or re-crawling content.
        """
        self.chunks.clear()
        self.embeddings = None