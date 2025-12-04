"""
Instance class for representing log sequences and their attributes
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


@dataclass
class Instance:
    """Class representing a log sequence instance with its attributes"""
    
    block_id: str
    log_sequence: List[Union[str, int]]
    label: str
    predicted: str = ""
    confidence: float = 0.0
    semantic_emb: Optional[Any] = None
    encode: Optional[Any] = None
    semantic_repr: List[Any] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed properties"""
        self.id = self.block_id
        self.sequence = self.log_sequence
        self.repr = None
    
    def __str__(self) -> str:
        """String representation of the instance"""
        sequence_str = " ".join(map(str, self.sequence))
        if not self.predicted:
            return f"{sequence_str}\n{self.id},{self.label}\n"
        return f"{sequence_str}\n{self.id},{self.label},{self.predicted},{self.confidence}\n"
    
    def __hash__(self) -> int:
        """Hash of the instance based on its string representation"""
        return int(hashlib.md5(str(self).encode("utf-8")).hexdigest(), 16)
    
    @property
    def seq_hash(self) -> int:
        """Hash of the sequence"""
        return hash(" ".join(map(str, self.sequence)))
    
    @property
    def event_count(self) -> Counter:
        """Count of events in the sequence"""
        return Counter(self.sequence)
