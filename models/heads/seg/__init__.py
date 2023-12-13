"""
Import segmentation head modules to guarantee their registration.
"""

from .drs import DRSHead  # noqa: F401
from .effseg import EffSegHead  # noqa: F401
from .roi import *  # noqa: F401, F403
