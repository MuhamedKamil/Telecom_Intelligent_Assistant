from collections import deque
from typing import List, Dict, Optional

class ChatMemory:
 
    def __init__(self, max_turns: int = 5):
        """
        Initialize chat memory.
        
        Args:
            max_turns: Maximum number of conversation turns to store
        """
        self._turns = deque(maxlen=max_turns)
        self.max_turns = max_turns
    
    def add_turn(self, user: str, assistant: str) -> None:
        """
        Add a conversation turn.
        
        Args:
            user: User message
            assistant: Assistant response
        """
        self._turns.append({
            "user": user,
            "assistant": assistant,
        })
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history in chat format for LLM.
        
        Returns:
            List of messages with "role" and "content" keys
            Example: [{"role": "user", "content": "..."}, ...]
        """
        messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages
    
    def get_last_turn(self) -> Optional[Dict[str, str]]:
        """
        Get the most recent conversation turn.
        
        Returns:
            Dict with "user" and "assistant" keys, or None if empty
        """
        if not self._turns:
            return None
        return self._turns[-1]
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
    
    def is_empty(self) -> bool:
        """Check if there is no conversation history."""
        return len(self._turns) == 0
    
    def __len__(self) -> int:
        """Return the number of stored conversation turns."""
        return len(self._turns)
    
    def __repr__(self) -> str:
        return f"ChatMemory(turns={len(self._turns)}, max_turns={self.max_turns})"