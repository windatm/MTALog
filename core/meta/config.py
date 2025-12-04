#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for source weighting
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SourceWeightingConfig:
    """Configuration for source weighting mechanism.
    
    Attributes:
        mode: Scoring mode - "none" (uniform), "distance_only", or "reliability_plus_distance"
        w_r: Weight for reliability component in scoring
        w_D: Weight for distance component in scoring
        beta_margin: Weight for margin in reliability computation
        kappa: Temperature parameter for softmax in α_s computation
    """
    mode: Literal["none", "distance_only", "reliability_plus_distance"] = "reliability_plus_distance"
    w_r: float = 1.0
    w_D: float = 1.0
    beta_margin: float = 0.1
    kappa: float = 1.0

