"""Runtime-aware logging utilities for :mod:`geosam`.

The default backend writes through Python's :mod:`logging` handlers. When the
runtime backend is configured as QGIS, messages are routed to the QGIS message
log so plugin users can see them from the QGIS interface.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geosam.context import get_runtime

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback

    class _TqdmFallback:
        """Minimal tqdm fallback used when tqdm is unavailable."""

        @staticmethod
        def write(message: str, file: Any = None) -> None:
            """Write a message without progress-bar support."""
            print(message, file=file or sys.stdout)

    tqdm = _TqdmFallback

if TYPE_CHECKING:
    from os import PathLike

__all__ = [
    "SUCCESS",
    "GeosamLogger",
    "LogLevel",
    "RuntimeLoggingHandler",
    "formatter",
    "get_default_log_level",
    "setup_logger",
    "stream_handler",
    "tqdm_handler",
]

SUCCESS: Literal[25] = 25
LogLevel = Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

logging.addLevelName(SUCCESS, "SUCCESS")


class GeosamLogger(logging.Logger):
    """Logger with a GeoSAM-specific success level."""

    def success(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Log a message with SUCCESS level.

        Parameters
        ----------
        message : object
            Message to log.
        *args : Any
            Positional logging arguments.
        **kwargs : Any
            Keyword logging arguments.

        """
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kwargs)


logging.setLoggerClass(GeosamLogger)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RuntimeLoggingHandler(logging.Handler):
    """Logging handler that dispatches records to the active runtime backend."""

    def __init__(self, stream: Any = sys.stdout) -> None:
        """Initialize the runtime logging handler.

        Parameters
        ----------
        stream : Any, optional
            Native Python output stream.

        """
        super().__init__()
        self.stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with the active backend."""
        try:
            message = self.format(record)
            if get_runtime().backend == "qgis":
                self._emit_qgis(record, message)
                return
            print(message, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # pragma: no cover - logging safety net
            self.handleError(record)

    def _emit_qgis(self, record: logging.LogRecord, message: str) -> None:
        """Emit a log record to QGIS message log."""
        from qgis.core import Qgis, QgsMessageLog

        if record.levelno >= logging.ERROR:
            level = Qgis.MessageLevel.Critical
        elif record.levelno >= logging.WARNING:
            level = Qgis.MessageLevel.Warning
        else:
            level = Qgis.MessageLevel.Info
        QgsMessageLog.logMessage(message, "GeoSAM", level)


class TqdmLoggingHandler(RuntimeLoggingHandler):
    """A runtime-aware logging handler that respects tqdm in native mode."""

    def __init__(self, tqdm_class: type[tqdm] = tqdm) -> None:  # type: ignore[assignment]
        """Initialize the tqdm logging handler.

        Parameters
        ----------
        tqdm_class : type[tqdm], optional
            Class used for progress-aware native writes.

        """
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using QGIS or tqdm-aware native output."""
        if get_runtime().backend == "qgis":
            super().emit(record)
            return
        try:
            message = self.format(record)
            self.tqdm_class.write(message, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # pragma: no cover - logging safety net
            self.handleError(record)


stream_handler = RuntimeLoggingHandler(sys.stdout)
stream_handler.setFormatter(formatter)

tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setLevel(logging.INFO)
tqdm_handler.setFormatter(formatter)


def get_default_log_level() -> int:
    """Get the default log level from GeoSAM environment settings.

    Returns
    -------
    int
        Python logging level.

    """
    log_level_str = os.getenv("GEOSAM_LOG_LEVEL", "").upper()
    if not log_level_str:
        log_level_str = os.getenv("FANINSAR_LOG_LEVEL", "").upper()
    if log_level_str in {"DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}:
        return getattr(logging, log_level_str if log_level_str != "SUCCESS" else "INFO")

    debug_flag = os.getenv("GEOSAM_DEBUG", os.getenv("FANINSAR_DEBUG", "0"))
    if debug_flag.lower() in {"1", "true", "yes", "on"}:
        return logging.DEBUG

    dev_indicators = [
        os.getenv("ENVIRONMENT") in ("dev", "development", "local"),
        os.getenv("ENV") in ("dev", "development", "local"),
        os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "on"},
        sys.argv and sys.argv[0].endswith(("pytest", "python", "ipython")),
        any("test" in arg for arg in sys.argv),
        hasattr(sys, "ps1"),
    ]
    if any(dev_indicators):
        return logging.DEBUG
    return logging.INFO


def setup_logger(
    name: str | None = None,
    file: str | PathLike[str] | None = None,
    *,
    log_name: str | None = None,
    log_file: str | PathLike[str] | None = None,
    handler: logging.Handler | list[logging.Handler] = stream_handler,
    level: int | None = None,
    propagate: bool = True,
    clear_existing: bool = False,
) -> GeosamLogger:
    """Create and configure a runtime-aware logger.

    Parameters
    ----------
    name : str | None, optional
        Logger name. If omitted, ``"geosam"`` is used.
    file : str | PathLike[str] | None, optional
        Optional file destination.
    log_name : str | None, optional
        Backward-compatible alias for ``name``.
    log_file : str | PathLike[str] | None, optional
        Backward-compatible alias for ``file``.
    handler : logging.Handler | list[logging.Handler], optional
        Handler or handlers to attach.
    level : int | None, optional
        Handler level. Defaults to :func:`get_default_log_level`.
    propagate : bool, optional
        Whether records propagate to parent loggers.
    clear_existing : bool, optional
        Whether existing handlers are cleared first.

    Returns
    -------
    GeosamLogger
        Configured logger instance.

    """
    logger_name = name or log_name or "geosam"
    file_target = file if file is not None else log_file
    effective_level = get_default_log_level() if level is None else level
    if not isinstance(effective_level, int) or effective_level < 0:
        message = f"Invalid logging level: {level}"
        raise ValueError(message)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate
    if clear_existing:
        logger.handlers.clear()

    handlers = [handler] if isinstance(handler, logging.Handler) else list(handler)
    for log_handler in handlers:
        if not isinstance(log_handler, logging.Handler):
            message = f"Expected logging.Handler, got {type(log_handler)}"
            raise TypeError(message)
        log_handler.setLevel(effective_level)
        if log_handler.formatter is None:
            log_handler.setFormatter(formatter)

    if file_target is not None:
        file_path = Path(file_target)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    existing_ids = {id(existing_handler) for existing_handler in logger.handlers}
    for log_handler in handlers:
        if id(log_handler) not in existing_ids:
            logger.addHandler(log_handler)

    return logger  # type: ignore[return-value]
