#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for managing random seed
"""

import random
import numpy as np
import torch
from typing import Optional

class RandomSeedManager:
    """Class to manage random seed for reproducibility"""
    
    @staticmethod
    def set_seed(seed: Optional[int] = 42) -> None:
        """
        Set seed for all random number generators
        
        Args:
            seed: Random seed value. If None, no seed will be set
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False 