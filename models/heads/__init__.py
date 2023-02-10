"""
Import head modules to guarantee their registration.
"""

from .box2d import BaseBox2dHead  # noqa: F401
from .cls import BaseClsHead  # noqa: F401
from .dino import DinoHead  # noqa: F401
from .dod import DOD  # noqa: F401
from .gvd import GVD  # noqa: F401
from .retina import RetinaHead, RetinaPredHead  # noqa: F401
from .roi import StandardRoIHead  # noqa: F401
from .sbd import SBD  # noqa: F401
from .seg import BaseSegHead, TopDownSegHead  # noqa: F401
