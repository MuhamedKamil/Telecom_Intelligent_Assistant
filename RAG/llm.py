import torch
from transformers import pipeline
from typing import List, Dict, Optional
from pathlib import Path
class LLM:
    """
    Simple LLM wrapper for text generation with context.
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict] 
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
        self.model_name        = llm_config["model_name"]
        self.max_context_chars = llm_config["max_context_chars"]
        self.system_prompt     = self._load_system_prompt(llm_config["system_prompt"]) 
        self.max_new_tokens    = llm_config["max_new_tokens"]
        self.temperature       = llm_config["temperature"]
        
        self.generator = pipeline(
            task         = "text-generation",
            model        = self.model_name,
            device_map   = "auto",
            torch_dtype  = torch.float16,

        )
        
        self.generation_kwargs = {
            "max_new_tokens"   : self.max_new_tokens,
            "do_sample"        : self.temperature > 0,
            "temperature"      : self.temperature if self.temperature > 0 else None,
            "return_full_text" : False,
            "eos_token_id"     : self.generator.tokenizer.eos_token_id,
        }
        
        self.generation_kwargs = {k: v for k, v in self.generation_kwargs.items() if v is not None}
    

    
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

    def _load_system_prompt(self, system_prompt: Optional[str]) -> str:
        """
        Load system prompt from the specified txt file.
        
        Args:
            system_prompt: Path to the .txt file containing the system prompt
            
        Returns:
            str: The system prompt content
        """
        # Load from file
        if system_prompt and isinstance(system_prompt, str) and system_prompt.endswith('.txt'):
            file_path = Path(system_prompt)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            return content
                        else:
                            print(f"Warning: System prompt file '{system_prompt}' is empty")
                except Exception as e:
                    print(f"Error reading prompt file '{system_prompt}': {e}")
            else:
                print(f"Warning: System prompt file '{system_prompt}' not found")
            
            # Fallback: Try to create the file with a default prompt
            print(f"Creating default system prompt file: {system_prompt}")
            default_prompt = "You are a helpful assistant for Telecom Egypt (WE). Answer questions using ONLY the provided context."
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            return default_prompt
        
        # If it's not a file path, assume it's the prompt string
        return system_prompt or "You are a helpful assistant. Answer questions using ONLY the provided context."