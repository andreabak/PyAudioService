"""Package with classes and modules related to audio acquisition, playback and recording"""

from .__version__ import __version__

from . import datatypes
from . import service
from . import record
from .datatypes import *
from .record import *
from .service import *


__all__ = datatypes.__all__ + service.__all__ + record.__all__
