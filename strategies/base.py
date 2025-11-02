"""Base strategy interface for category handlers."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class CategoryStrategy(ABC):
    """Abstract base class for category strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the category name."""
        pass
    
    @property
    @abstractmethod
    def instruction(self) -> str:
        """Return the instruction for generating questions in this category."""
        pass
    
    @abstractmethod
    def build_generation_messages(self, system_prompt: str) -> list:
        """
        Build messages for question generation.
        
        Args:
            system_prompt: System prompt to use
            
        Returns:
            List of message dicts
        """
        pass

