import torch
from typing import List, Dict, Any, Optional

class WebsiteStore:
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

    def load(self, chunks: List[Dict[str, Any]], embeddings: torch.Tensor) -> None:
        """Load chunks and their embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
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
        self.chunks.clear()
        self.embeddings = None