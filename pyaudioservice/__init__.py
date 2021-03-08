"""Package with classes and modules related to audio acquisition, playback and recording"""

__version__ = "0.1.1"

from . import datatypes
from . import service
from . import record
from .datatypes import *
from .record import *
from .service import *


__all__ = datatypes.__all__ + service.__all__ + record.__all__
