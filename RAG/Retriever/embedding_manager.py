import torch
from FlagEmbedding import BGEM3FlagModel
from typing import List, Union,Optional,Dict

class EmbeddingManager:
    """
    Generates dense vector embeddings for documents and user queries to power semantic search over 
    the knowledge base
    """
    def __init__(
        self,
        # model_name: str = "BAAI/bge-m3",
        # device: str     = None,
        # use_fp16: bool  = True,
        # batch_size: int = 16,
        # max_length: int = 8192,
        embedder_config: Optional[Dict] 

    ):
        
        self.model_name = embedder_config["model_name"]
        self.use_fp16   = embedder_config["use_fp16"]
        self.device     = embedder_config["device"]
        self.batch_size = embedder_config["batch_size"]
        self.max_length = embedder_config["max_length"]


        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16 = self.use_fp16 and device == self.device,
            device   = self.device,
        )

    def embed_documents(self, texts: List[str]) -> torch.Tensor:
        """
        Convert a list of text documents into dense vector embeddings.
        
        Args:
            texts (List[str]): List of text strings to embed.
            
        Returns:
            torch.Tensor: Tensor of shape (len(texts), embedding_dim) containing 
                         the dense vector representations.
        """
        
        embeddings = self.model.encode(
            texts,
            batch_size  = self.batch_size,
            max_length  = self.max_length,
        )["dense_vecs"]
        return torch.tensor(embeddings)

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Embed a single query string for similarity search.
        
        Args:
            query (str): The query text to embed.
            
        Returns:
            torch.Tensor: Tensor of shape (embedding_dim,) containing the query embedding.
        """        
        return self.embed_documents([query])[0]

    @staticmethod
    def similarity(query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine/ dot-product similarity between a query and multiple documents.
        
        Args:
            query_embedding (torch.Tensor): Query embedding of shape (embedding_dim,).
            document_embeddings (torch.Tensor): Document embeddings of shape 
                                               (n_documents, embedding_dim).
            
        Returns:
            torch.Tensor: Similarity scores of shape (n_documents,) where higher values
                         indicate more relevant documents.
        """
        return query_embedding @ document_embeddings.T


