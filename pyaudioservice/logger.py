"""Logging setup and customized logger code"""

import logging
from copy import copy
from threading import RLock
from typing import MutableMapping, Any, Tuple, Type, Optional, Union, Callable

import atexit

from blessed import Terminal


__all__ = ['log', 'set_log_level', 'custom_log', 'LoggerType']  # Export logger


LoggerType = Union[logging.Logger, logging.LoggerAdapter]


term = Terminal()

STYLE_RESET: str = term.normal
LOG_FORMAT: str = '%(asctime)s [%(component)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT: str = '%Y-%m-%d %H:%M:%S'

atexit.register(lambda: print(STYLE_RESET))

logging_lock: RLock = RLock()


class ColorFormatter(logging.Formatter):
    """
    Custom log formatter class to colorize log levels in console output
    """
    LOG_COLORS: MutableMapping[str, int] = {
        'NOTSET': term.gray33,
        'DEBUG': term.steelblue4,
        'INFO': term.forestgreen,
        'WARNING': term.yellow3,
        'ERROR': term.orangered3,
        'CRITICAL': term.red,
    }

    def __init__(self, fmt, *args, **kwargs):
        fmt: str = f'{STYLE_RESET}%(log_color)s{fmt}{STYLE_RESET}'
        super().__init__(fmt, *args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.LOG_COLORS:
            record.__dict__['log_color'] = self.LOG_COLORS[record.levelname]
        return super().format(record)


class CustomLoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter to allow project-specific extra data to be passed to logging function calls
    """
    LOG_EXTRA_DATA: MutableMapping[str, str] = {'component': 'MAIN'}

    def __init__(self, logger: logging.Logger, **extra_data: Any):
        adapter_extra_data: MutableMapping[str, Any] = copy(self.LOG_EXTRA_DATA)
        adapter_extra_data.update(extra_data)
        super().__init__(logger, adapter_extra_data)

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> Tuple[Any, MutableMapping[str, Any]]:
        old_extra: MutableMapping[str, Any] = kwargs.get('extra', None) or {}
        new_extra: MutableMapping[str, Any] = dict(self.extra)
        # Inject custom kwargs
        if 'component' in kwargs:
            new_extra['component'] = kwargs.pop('component')
        new_extra.update(old_extra)
        kwargs['extra'] = new_extra
        return msg, kwargs

    def log(self, *args, **kwargs: Any) -> None:
        global logging_lock
        with logging_lock:
            super().log(*args, **kwargs)


def custom_log(optional_cls: Optional[Type] = None, **logger_params) -> Union[Type, Callable[[Type], Type]]:
    """
    A class decorator that adds customized logging functionality to the decorated class
    through a `.logger` attribute and with a private dunder attribute '.__log'
    :param optional_cls: The class passed by python to be decorated
    :param logger_params: Additional default format parameters for the logger adapter
    """
    def decorator(cls: Type) -> Type:
        global _bare_logger
        nonlocal logger_params

        logger: CustomLoggerAdapter = CustomLoggerAdapter(_bare_logger, **logger_params)
        setattr(cls, 'logger', logger)
        setattr(cls, f'_{cls.__name__}__log', logger)

        return cls

    if optional_cls is None:
        return decorator
    else:
        return decorator(optional_cls)


# Setup the log for the project
_bare_logger: logging.Logger = logging.getLogger('WesternAmber')
log_handler: logging.StreamHandler = logging.StreamHandler()
log_handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=LOG_DATEFORMAT))
_bare_logger.addHandler(log_handler)
_bare_logger.setLevel(logging.INFO)
log: CustomLoggerAdapter = CustomLoggerAdapter(_bare_logger)


def set_log_level(level: str) -> None:
    """
    Set the log level for the base logger
    :param level: the level to set for the logger as a string
    """
    global _bare_logger
    _bare_logger.setLevel(level)

# TODO: Consider installing ColorFormatter with a root logger StreamHandler to capture libraries logs too
