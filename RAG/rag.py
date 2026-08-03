import torch
from typing import List, Dict, Optional
from .Retriever import (
    EmbeddingManager,
    WebsiteStore,
    UploadedStore,
    RetrievalManager,
)
from .llm import LLM
from .memory import ChatMemory

class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system that combines document retrieval with LLM generation.
    
    This class orchestrates the entire RAG pipeline including embedding generation,
    document retrieval from multiple sources (website and uploaded), and response
    generation using a language model with contextual information.
    
    """

    def __init__(
        self,
        Rag_config: Optional[Dict] 

    ):
        self.embedder_config: str         = Rag_config["embedder"]
        self.llm_config                   = Rag_config["llm"]
        self.max_turns: int               = Rag_config["memory"]["max_turns"]
        self.top_k: int                   = Rag_config["generation"]["top_k"]
        
        """
        Initialize the RAG system with specified models and configuration.
        
        Args:
            embedder_model (str, optional): HuggingFace model name for embeddings.
                Defaults to "BAAI/bge-m3".
            device (Optional[str], optional): Device to run models on ("cpu", "cuda", "mps").
                If None, auto-detects best available device. Defaults to None.
            use_fp16 (bool, optional): Whether to use half-precision (FP16) for inference.
                Reduces memory usage and speeds up computation. Defaults to True.
            llm_model (str, optional): HuggingFace model name for the language model.
                Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            max_turns (int, optional): Maximum conversation turns to keep in context.
                Older turns are dropped to prevent context overflow. Defaults to 5.
            top_k (int, optional): Number of relevant document chunks to retrieve.
                Higher values provide more context but may include less relevant info.
                Defaults to 5.
            max_new_tokens (int, optional): Maximum tokens to generate per response.
                Controls response length. Defaults to 512.
            system_prompt (Optional[str], optional): Custom system prompt to guide
                the LLM's behavior. If None, uses default RAG-specific prompt.
                Defaults to None.
        """
        self.embedding_manager = EmbeddingManager(
            embedder_config    = self.embedder_config
        )
        self.website_store     = WebsiteStore()
        self.uploaded_store    = UploadedStore()
        
        self.retrieval_manager = RetrievalManager(
            embedding_manager  = self.embedding_manager,
            website_store      = self.website_store,
            uploaded_store     = self.uploaded_store,
        )
        
        self.llm = LLM(
            llm_config = self.llm_config
        )
        
        self.memory = ChatMemory(max_turns=self.max_turns)
    
    def add_website_documents(
        self,
        chunks: List[Dict],
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add website documents to the retriever.
        
        Args:
            chunks     : List of document chunks with metadata
            embeddings : Pre-computed embeddings (optional)
        """
        if embeddings is None:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_manager.embed_documents(texts)
        
        self.website_store.load(chunks, embeddings)
    
    def add_uploaded_documents(
        self,
        document: Dict,
        chunks: List[Dict],
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add uploaded documents to the retriever.
        
        Args:
            document   : Document metadata
            chunks     : List of document chunks with metadata
            embeddings : Pre-computed embeddings (optional)
        """
        if embeddings is None:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_manager.embed_documents(texts)
        
        self.uploaded_store.index(document, chunks, embeddings)
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get a response.
        
        Args:
            question: User question
            
        Returns:
            Generated response string
        """
        retrieved_chunks = self.retrieval_manager.retrieve(
            query=question,
            top_k=self.top_k,
        )
        context_chunks = [chunk["text"] for chunk in retrieved_chunks]
        history        = self.memory.get_history()
        response       = self.llm.generate(
            question       = question,
            context_chunks = context_chunks,
            chat_history   = history,
        )
        self.memory.add_turn(question, response)
        
        return response
    
    def ask_with_sources(self, question: str) -> Dict:
        """
        Ask a question and get response with source information.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with response and source documents
        """
        retrieved_chunks = self.retrieval_manager.retrieve(
            query = question,
            top_k = self.top_k,
        )
        
        context_chunks = []
        sources        = []
        for chunk in retrieved_chunks:
            context_chunks.append(chunk["text"])
            sources.append({
                "text"        : chunk["text"],
                "source"      : chunk.get("source", "unknown"),
                "score"       : chunk.get("score", 0.0),
                "title"       : chunk.get("title", "Untitled"),
                "url"         : chunk.get("url", "N/A"),         
                "chunk_id"    : chunk.get("chunk_id", ""),
                "chunk_index" : chunk.get("chunk_index", 0),
                "language"    : chunk.get("language", "unknown"),
            })
        
        history = self.memory.get_history()
        
        response = self.llm.generate(
            question       = question,
            context_chunks = context_chunks,
            chat_history   = history,
        )
        
        self.memory.add_turn(question, response)
        
        return {
            "question"     : question,
            "response"     : response,
            "sources"      : sources,
            "total_sources": len(sources),
        }
        
    def clear(self) -> None:
        """Clear all data (documents and memory)."""
        self.website_store.clear()
        self.uploaded_store.clear()
        self.memory.clear()
    
    def clear_memory(self) -> None:
        """Clear only conversation memory."""
        self.memory.clear()
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.memory.get_history()
    
    @property
    def document_count(self) -> int:
        """Get total number of documents in both stores."""
        website_count = len(self.website_store.chunks)
        uploaded_count = len(self.uploaded_store.chunks)
        return website_count + uploaded_count
    
    def __repr__(self) -> str:
        return f"RAGSystem(documents={self.document_count}, turns={len(self.memory)})"

