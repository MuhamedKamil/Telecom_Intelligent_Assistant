import torch
from typing import List, Dict, Any, Optional

class UploadedStore:
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

    def index(self, document: Dict[str, Any], chunks: List[Dict[str, Any]], embeddings: torch.Tensor) -> None:
        """Index a document with its chunks and embeddings."""
        self.documents.append(document)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

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
            chunk["source"] = "uploaded"
            results.append(chunk)

        return results

    def is_empty(self) -> bool:
        """Check if store is empty."""
        return len(self.chunks) == 0

    def clear(self) -> None:
        """Clear all stored data."""
        self.documents.clear()
        self.chunks.clear()
        self.embeddings = None