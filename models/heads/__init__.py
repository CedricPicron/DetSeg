"""
Import head modules to guarantee their registration.
"""

from .binary import BinarySegHead  # noqa: F401
from .brd import BRD  # noqa: F401
from .box2d import BaseBox2dHead  # noqa: F401
from .cls import BaseClsHead  # noqa: F401
from .dfd import DFD  # noqa: F401
from .dod import DOD  # noqa: F401
from .gvd import GVD  # noqa: F401
from .mbd import MBD  # noqa: F401
from .retina import RetinaHead, RetinaPredHead  # noqa: F401
from .sbd import SBD  # noqa: F401
from .semantic import SemanticSegHead  # noqa: F401
