"""
Import head modules to guarantee their registration.
"""

from .bde import BDE  # noqa: F401
from .box2d import BaseBox2dHead  # noqa: F401
from .cls import BaseClsHead  # noqa: F401
from .drs import DRSHead  # noqa: F401
from .dup import BaseDuplicateHead  # noqa: F401
from .effseg import EffSegHead  # noqa: F401
from .gvd import GVD  # noqa: F401
from .roi import StandardRoIHead  # noqa: F401
