"""
Import segmentation head modules to guarantee their registration.
"""

from .drs import DRSHead  # noqa: F401
from .effseg import EffSegHead  # noqa: F401
from .mmdet_roi import *  # noqa: F401, F403
from .mod_roi import ModRoIHead  # noqa: F401
