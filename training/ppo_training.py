# -*- coding: utf-8 -*-
"""Compatibility entry for the historical filename; the implementation is RLOO.

The original local file already used RLOOTrainer despite its PPO filename.
Use `python -m training.rloo_training` for unambiguous new experiments.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rloo_training import main

if __name__ == '__main__':
    main()
