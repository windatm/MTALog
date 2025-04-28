#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary"""
        return {
            "id": self.id,
            "sequence": self.sequence,
            "label": self.label,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "semantic_emb": self.semantic_emb,
            "encode": self.encode,
            "semantic_repr": self.semantic_repr,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Instance':
        """Create instance from dictionary"""
        return cls(
            block_id=data["id"],
            log_sequence=data["sequence"],
            label=data["label"],
            predicted=data.get("predicted", ""),
            confidence=data.get("confidence", 0.0),
            semantic_emb=data.get("semantic_emb"),
            encode=data.get("encode"),
            semantic_repr=data.get("semantic_repr", []),
        )
