"""Tennis Analytics - Professional Tennis Video Analysis System"""

__version__ = "2.0.0"
__author__ = "Tennis Analytics Team"

# Make key components easily importable
from . import core
from . import detection
from . import analysis
from . import training
from . import evaluation

__all__ = [
    'core',
    'detection',
    'analysis',
    'training',
    'evaluation',
]

