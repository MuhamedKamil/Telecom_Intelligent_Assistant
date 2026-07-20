from typing import List, Dict, Any, Optional
from .embedding_manager import EmbeddingManager
from .website_store import WebsiteStore
from .uploaded_store import UploadedStore

class RetrievalManager:
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        website_store: WebsiteStore,
        uploaded_store: UploadedStore,
    ):
        self.embedding_manager = embedding_manager
        self.website_store = website_store
        self.uploaded_store = uploaded_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from all stores."""
        query_embedding = self.embedding_manager.embed_query(query)
        
        results = []
        
        # Search website store
        results.extend(self.website_store.search(query_embedding, top_k=top_k))
        
        # Search uploaded store if not empty
        if not self.uploaded_store.is_empty():
            results.extend(self.uploaded_store.search(query_embedding, top_k=top_k))
        
        # Sort and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]