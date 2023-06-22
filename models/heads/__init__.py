"""
Import head modules to guarantee their registration.
"""

from .bde import BDE  # noqa: F401
from .box2d import BaseBox2dHead  # noqa: F401
from .cls import BaseClsHead  # noqa: F401
from .dup import BaseDuplicateHead  # noqa: F401
from .gvd import GVD  # noqa: F401
from .roi import StandardRoIHead  # noqa: F401
from .seg import BaseSegHead, TopDownSegHead  # noqa: F401
