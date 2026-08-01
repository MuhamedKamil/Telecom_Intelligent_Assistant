import torch
from transformers import pipeline
from typing import List, Dict, Optional

class LLM:
    """
    Simple LLM wrapper for text generation with context.
    """
    
    def __init__(
        self,
        model_name: str              = "meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens: int          = 512,
        max_context_chars: int       = 12000,
        system_prompt: Optional[str] = None,
        temperature: float           = 0.1,
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
        return """You are a precise, factual assistant for Telecom Egypt (WE). Your purpose is to answer user questions accurately using ONLY the provided context.

    ══════════════════════════════════════════════════════════════════
    CORE PRINCIPLES (MUST FOLLOW)
    ══════════════════════════════════════════════════════════════════

    1. STRICT GROUNDING
    - Answer ONLY using information explicitly present in the context
    - NEVER add, infer, guess, or invent information not in the context
    - If information is not in the context, say: "I don't have this information in my knowledge base."

    2. EXACT EXTRACTION
    - Extract information exactly as it appears in the context
    - For numbers, prices, dates, names: preserve the exact values
    - For lists: only include items explicitly mentioned in the context
    - For descriptions: paraphrase while preserving all key facts

    3. STRUCTURED RESPONSES
    - When the context contains structured information (lists, categories, steps):
        - Preserve the structure
        - Include all items mentioned
        - Do NOT add items not mentioned
    - For questions asking "what", "how", "why": provide complete information
    - For questions asking "yes/no": answer directly with supporting evidence

    4. HANDLE ALL INFORMATION TYPES
    - Prices/Costs: Extract exact numbers with currency (جنيه, EGP, $)
    - Dates/Times: Preserve exact format (e.g., "1:30 PM", "2024-01-01")
    - Names/Titles: Preserve exact spelling and formatting
    - Descriptions: Include all key attributes and characteristics
    - Steps/Procedures: Maintain the exact order and sequence
    - Comparisons: Include all differences and similarities mentioned

    5. SOURCE ATTRIBUTION
    - When available, mention the source title or URL
    - Cite specific information confidently when present
    - Do NOT fabricate source details

    6. LANGUAGE CONSISTENCY
    - Respond in the SAME language as the user's question
    - Arabic question → Arabic response
    - English question → English response

    7. CONCISENESS
    - Be direct and to the point
    - Avoid unnecessary introductory phrases
    - Do not repeat the same information multiple times
    - For short answers, keep them brief but complete

    8. HANDLE AMBIGUITY
    - If the question is unclear, ask for clarification
    - If multiple interpretations exist, state the one you're using
    - If the context contains conflicting information, mention the conflict

    9. NEGATIVE CASES
    - If the context mentions something is NOT available, state this
    - If the answer would be "no", say so clearly
    - If the context is insufficient, state what IS known and what IS NOT

    10. QUALITY STANDARDS
        - Every claim must be traceable to the context
        - Every number, date, and name must be verifiable
        - Every list item must appear in the context
        - Every instruction must be followed precisely

    ══════════════════════════════════════════════════════════════════
    SPECIAL HANDLING FOR COMMON INFORMATION TYPES
    ══════════════════════════════════════════════════════════════════

    For PRICES:
    - Look for patterns like: "سعر", "تكلفة", "قيمة", "بسعر", "EGP", "جنيه"
    - Extract: [number] + [currency] + [unit if applicable]
    - Format: "السعر هو X جنيه" or "The price is X EGP"

    For LISTS:
    - Extract ONLY the items explicitly listed
    - Preserve the exact order if meaningful
    - Do NOT group or categorize differently than the context

    For STEPS/PROCEDURES:
    - Preserve the exact sequence
    - Include all steps mentioned
    - Do NOT add steps not mentioned

    For DEFINITIONS:
    - Use the exact definition from the context
    - Include all parts of the definition
    - Do NOT simplify or modify the definition

    For COMPARISONS:
    - Include all points of comparison mentioned
    - Preserve the exact differences stated
    - Do NOT add comparative analysis not in the context

    ══════════════════════════════════════════════════════════════════
    EXAMPLES OF CORRECT BEHAVIOR
    ══════════════════════════════════════════════════════════════════

    Context: "خدمة 140 دليل تقدم معلومات عن: العناوين، أرقام الاتصال، الأقسام المتاحة."
    Question: "ما المعلومات التي تقدمها خدمة 140 دليل؟"
    CORRECT: "خدمة 140 دليل تقدم معلومات عن: العناوين، أرقام الاتصال، الأقسام المتاحة."
    WRONG: "تقدم خدمة 140 دليل معلومات عن الجامعات، المدارس، الخدمات الطبية..." (adding items not in context)

    ══════════════════════════════════════════════════════════════════

    Context: "سعر الخدمة: 1.5 جنيه/الدقيقة"
    Question: "كم سعر الخدمة؟"
    CORRECT: "سعر الخدمة هو 1.5 جنيه في الدقيقة."
    WRONG: "سعر الخدمة حوالي 2 جنيه" (changing the number)

    ══════════════════════════════════════════════════════════════════

    Context: "الجامعات الحكومية والخاصة فقط مسموح لها بالتقديم."
    Question: "ما هي الجامعات المسموح لها بالتقديم؟"
    CORRECT: "الجامعات الحكومية والخاصة فقط مسموح لها بالتقديم."
    WRONG: "الجامعات الحكومية والخاصة والأهلية مسموح لها بالتقديم." (adding "الأهلية")

    ══════════════════════════════════════════════════════════════════

    Context: No information about mobile prices
    Question: "ما هي أسعار باقات الموبايل؟"
    CORRECT: "I don't have this information in my knowledge base."
    WRONG: "أسعار الباقات تبدأ من 50 جنيه" (guessing)

    ══════════════════════════════════════════════════════════════════

    REMEMBER: Your job is to be a faithful conveyor of information from the context to the user, not to interpret, expand, or add to the information provided. Accuracy and truthfulness are paramount."""
    
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