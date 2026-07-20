import torch
from FlagEmbedding import BGEM3FlagModel
from typing import List, Union

class EmbeddingManager:
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 8192,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device     = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and device == "cuda",
            device=device,
        )

    def embed_documents(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        return torch.tensor(embeddings)

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query."""
        return self.embed_documents([query])[0]

    @staticmethod
    def similarity(query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and document embeddings."""
        return query_embedding @ document_embeddings.T