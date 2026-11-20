"""ASE calculator wrapper for g-xTB."""

from .calculator import GxTB
from .v2 import GxTB2

__version__ = "0.1.0"
__author__ = "Stefan Bringuier"
__email__ = "stefanbringuier@gmail.com"
__description__ = "ASE calculator wrapper for g-xTB"

__all__ = ["GxTB", "GxTB2"]
