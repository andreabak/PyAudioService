"""Package with classes and modules related to audio acquisition, playback and recording"""

from . import audio
from . import record
from .record import *
from .audio import *


__all__ = audio.__all__ + record.__all__
