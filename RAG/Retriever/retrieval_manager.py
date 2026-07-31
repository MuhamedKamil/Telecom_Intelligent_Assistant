from typing import List, Dict, Any
from .embedding_manager import EmbeddingManager
from .website_store import WebsiteStore
from .uploaded_store import UploadedStore

class RetrievalManager:
    """
    Manages retrieval of relevant information from multiple data sources.
    
    This class orchestrates searching across website content and uploaded documents
    to find the most relevant chunks for a given query.
    """
    
    def __init__(
        self,
        embedding_manager : EmbeddingManager,
        website_store     : WebsiteStore,
        uploaded_store    : UploadedStore,
    ):
        """
            Initialize the retrieval manager with necessary components.
                
                Args:
                    embedding_manager (EmbeddingManager): Manager for generating query embeddings.
                    website_store (WebsiteStore): Store containing indexed website content.
                    uploaded_store (UploadedStore): Store containing uploaded document content.
        """
        self.embedding_manager = embedding_manager
        self.website_store     = website_store
        self.uploaded_store    = uploaded_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks from all available stores.
        
        Args:
            query (str): The search query text.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of top-k relevant chunks, sorted by relevance score
                                  in descending order. Each dictionary contains chunk data
                                  and a similarity score.
        """
        
        query_embedding = self.embedding_manager.embed_query(query)
        results         = []
        
        results.extend(self.website_store.search(query_embedding, top_k=top_k))
        if not self.uploaded_store.is_empty():
            results.extend(self.uploaded_store.search(query_embedding, top_k=top_k))
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]