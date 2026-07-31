import torch
from typing import List, Dict, Any, Optional

class UploadedStore:
    """
    In-memory vector store for uploaded documents with semantic search capability.
    
    This class manages indexing, storing, and searching through uploaded documents
    by maintaining document metadata, text chunks, and their corresponding embeddings.
    """
    
    def __init__(self):
        self.documents  : List[Dict[str, Any]]   = []
        self.chunks     : List[Dict[str, Any]]   = []
        self.embeddings : Optional[torch.Tensor] = None

    def index(self, document: Dict[str, Any], chunks: List[Dict[str, Any]], embeddings: torch.Tensor) -> None:
        """
        Index a document by storing its metadata, chunks, and embeddings.
        
        Args:
            document (Dict[str, Any]): Document-level metadata (e.g., filename, upload date).
            chunks (List[Dict[str, Any]]): List of text chunks with their metadata.
            embeddings (torch.Tensor): Tensor of shape (num_chunks, embedding_dim) containing
                                      dense vector representations for each chunk.
        """

        self.documents.append(document)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:

        """
        Search for the most similar chunks to a query embedding.
        
        Args:
            query_embedding (torch.Tensor): Query embedding vector of shape (embedding_dim,).
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of top-k relevant chunks, each enriched with:
                - All original chunk metadata
                - 'score' (float): Similarity score (higher = more relevant)
                - 'source' (str): Always "uploaded" to identify the source
        """
        if not self.chunks:
            return []

        scores = query_embedding @ self.embeddings.T
        values, indices = torch.topk(scores, k = min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            chunk           = self.chunks[idx].copy()
            chunk["score"]  = score
            chunk["source"] = "uploaded"
            results.append(chunk)

        return results

    def is_empty(self) -> bool:
        """
        Check if the store contains any indexed chunks.
        
        Returns:
            bool: True if no chunks are stored, False otherwise.
        """
        return len(self.chunks) == 0

    def clear(self) -> None:
       """
        Clear all stored data from the store.
        This removes all documents, chunks, and embeddings to free memory.
        Useful when resetting the store or re-indexing content.
        """
       self.documents.clear()
       self.chunks.clear()
       self.embeddings = None