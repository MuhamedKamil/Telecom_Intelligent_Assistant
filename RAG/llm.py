import torch
from transformers import pipeline
from typing import List, Dict, Optional

class LLM:
    """
    Simple LLM wrapper for text generation with context.
    """
    
    def __init__(
        self,
        model_name: str              = "meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens: int          = 512,
        max_context_chars: int       = 12000,
        system_prompt: Optional[str] = None,
        temperature: float           = 0.0,
    ):
        """
        Initialize the LLM.
        
        Args:
            model_name: HuggingFace model name
            max_new_tokens: Maximum tokens to generate
            max_context_chars: Maximum context length in characters
            system_prompt: Custom system prompt (uses default if None)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model_name        = model_name
        self.max_context_chars = max_context_chars
        self.system_prompt     = system_prompt or self._default_system_prompt()
        
        self.generator = pipeline(
            task         = "text-generation",
            model        = model_name,
            device_map   = "auto",
            torch_dtype  = torch.float16,
        )
        
        self.generation_kwargs = {
            "max_new_tokens"   : max_new_tokens,
            "do_sample"        : temperature > 0,
            "temperature"      : temperature if temperature > 0 else None,
            "return_full_text" : False,
            "eos_token_id"     : self.generator.tokenizer.eos_token_id,
        }
        
        self.generation_kwargs = {k: v for k, v in self.generation_kwargs.items() if v is not None}
    
    @staticmethod
    def _default_system_prompt() -> str:
         return """You are a helpful assistant for Telecom Egypt (WE).
Answer questions using ONLY the provided context.
If the answer is not in the context, say "I don't have this information in my knowledge base."
DO NOT refer to documents by number like "Document 1" or "Document 4".
Simply provide the answer based on the information given.
Be concise and accurate."""
    
    def _build_context(self, context_chunks: List[str]) -> str:
        """
        Build context string from chunks, respecting character limit.
        
        Args:
            context_chunks: List of text chunks
            
        Returns:
            Formatted context string
        """
        context = ""
        for i, chunk in enumerate(context_chunks, 1):
            section = f"Document {i}\n----------\n{chunk.strip()}\n\n"
            if len(context) + len(section) > self.max_context_chars:
                break
            context += section
        return context.strip()
    
    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Build messages for the LLM.
        
        Args:
            question: User question
            context: Formatted context string
            chat_history: Optional conversation history
            
        Returns:
            List of messages in chat format
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
        
        return messages
    
    def generate(
        self,
        question: str,
        context_chunks: List[str],
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a response based on question and context.
        
        Args:
            question: User question
            context_chunks: List of relevant text chunks
            chat_history: Optional conversation history
            
        Returns:
            Generated response string
        """
        context  = self._build_context(context_chunks)
        messages = self._build_messages(question, context, chat_history)
        prompt   = self.generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.generator(prompt, **self.generation_kwargs)
        return outputs[0]["generated_text"].strip()
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the model name."""
        self._model_name = value