"""
Centralized Logging Configuration for the Telescope Controller.

Architecture Role
-----------------
This module is the **single logging configuration point** for the entire
application.  Every module should obtain its logger through::

    from telescope_logger import get_logger
    logger = get_logger(__name__)

The ``setup_logging()`` function (called once at application start) configures:
  - A **RotatingFileHandler** that captures everything at DEBUG level.
  - An optional **GUI callback handler** that forwards messages to the tkinter
    log panel with color-coded severity tags.
  - A **StreamHandler** (console) as a fallback when no GUI is present.

Tag System
----------
The GUI log panel uses a tag-based color scheme.  This module defines the
canonical mapping between Python logging levels and GUI tags:

    ============  ============  ===============
    Level         GUI Tag       Color
    ============  ============  ===============
    DEBUG         ``"cmd"``     blue (#4a9eff)
    INFO          ``"info"``    dim gray
    SUCCESS (25)  ``"success"`` orange accent
    WARNING       ``"warning"`` yellow-orange
    ERROR         ``"error"``   red
    CRITICAL      ``"error"``   red
    ============  ============  ===============

Sub-modules that previously used bare ``print()`` or ad-hoc ``on_log``
callbacks should migrate to ``get_logger(__name__)`` and call standard
``logger.info(...)``, ``logger.warning(...)``, etc.

Backward Compatibility
----------------------
The legacy ``TelescopeLogger`` class is preserved as a thin wrapper around
``get_logger()`` so existing code that instantiates it (e.g., in
main_realtime.py) continues to work without changes.

Usage Examples
--------------
**In any module** (no GUI dependency)::

    from telescope_logger import get_logger
    logger = get_logger(__name__)

    logger.info("Connected to mount on COM3")
    logger.warning("Plate-solve took %.0fms (slow)", solve_ms)
    logger.error("Serial port error: %s", err)

**At application startup** (main_realtime.py)::

    from telescope_logger import setup_logging, get_logger, set_gui_callback

    setup_logging(log_file="telescope_app.log")
    set_gui_callback(my_gui_log_function)   # optional
    logger = get_logger("main")
    logger.info("Application started")

Dependencies
------------
Python standard library only (``logging``, ``os``, ``pathlib``).
"""

import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, MemoryHandler
from typing import Optional, Callable, Dict

# ---------------------------------------------------------------------------
#  Custom log level: SUCCESS (between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")


def _success(self, message, *args, **kwargs):
    """Log a message at the SUCCESS level (25)."""
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)


# Monkey-patch Logger so every logger instance gets a .success() method
logging.Logger.success = _success  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
#  Module-level state
# ---------------------------------------------------------------------------
_root_logger_name = "TelescopeApp"
_is_configured = False

# Optional GUI callback: Callable[[str, str], None]  -> (message, tag)
_gui_callback: Optional[Callable[[str, str], None]] = None

# Map Python logging levels to GUI color-tag strings
LEVEL_TO_TAG: Dict[int, str] = {
    logging.DEBUG:    "cmd",
    logging.INFO:     "info",
    SUCCESS:          "success",
    logging.WARNING:  "warning",
    logging.ERROR:    "error",
    logging.CRITICAL: "error",
}

# Reverse map: GUI tag string -> Python logging level (for the legacy API)
TAG_TO_LEVEL: Dict[str, int] = {
    "info":     logging.INFO,
    "success":  SUCCESS,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "cmd":      logging.DEBUG,
    "rate":     logging.DEBUG,
    "tracking": logging.DEBUG,
    "server":   logging.DEBUG,
    "response": logging.DEBUG,
    "usb":      logging.DEBUG,
}


# ---------------------------------------------------------------------------
#  GUI Callback Handler
# ---------------------------------------------------------------------------
class _GUICallbackHandler(logging.Handler):
    """Logging handler that forwards records to the tkinter GUI log panel.

    The handler converts the Python log level to a GUI tag string and calls
    the registered callback with ``(formatted_message, tag)``.

    This handler is attached to the root TelescopeApp logger by
    ``set_gui_callback()`` and removed by ``remove_gui_callback()``.
    """

    def emit(self, record: logging.LogRecord):
        try:
            if _gui_callback is None:
                return
            msg = self.format(record)
            tag = LEVEL_TO_TAG.get(record.levelno, "info")

            # Allow overriding the tag via record.gui_tag (set by log_with_tag)
            tag = getattr(record, "gui_tag", tag)

            _gui_callback(msg, tag)
        except Exception:
            self.handleError(record)


# Singleton handler instance (so we can add/remove it cleanly)
_gui_handler: Optional[_GUICallbackHandler] = None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def setup_logging(
    log_file: str = "telescope_app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure the application-wide logging infrastructure.

    Call this **once** at application startup, before any ``get_logger()``
    calls.  It is safe to call multiple times (subsequent calls are no-ops).

    Args:
        log_file:      Path to the rotating log file.
        max_bytes:     Maximum file size before rotation (default 10 MB).
        backup_count:  Number of rotated backup files to keep.
        console_level: Minimum level for console (stderr) output.
        file_level:    Minimum level for file output.

    Returns:
        The root application logger.
    """
    global _is_configured
    if _is_configured:
        return logging.getLogger(_root_logger_name)

    root = logging.getLogger(_root_logger_name)
    root.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Prevent duplicate handlers on repeated imports / calls
    if root.handlers:
        _is_configured = True
        return root

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (INFO+ by default) ---
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- Rotating file handler (DEBUG+) ---
    # On Linux / Android, wrap the file handler in a MemoryHandler that
    # batches up to 64 records before flushing to disk.  This dramatically
    # reduces write operations when the 5Hz tracking loop generates
    # frequent DEBUG messages.  Any WARNING or above triggers an immediate
    # flush so errors are never lost.  On Windows the
    # RotatingFileHandler is used directly (disk I/O is fast enough).
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(file_level)
        fh.setFormatter(formatter)

        if sys.platform != "win32":
            # Batch writes: buffer up to 64 records, flush on WARNING+
            mh = MemoryHandler(
                capacity=64,
                flushLevel=logging.WARNING,
                target=fh,
                flushOnClose=True,
            )
            mh.setLevel(file_level)
            root.addHandler(mh)
        else:
            root.addHandler(fh)
    except Exception as exc:
        # Cannot create file handler -- log to console only
        root.warning("Could not create log file handler: %s", exc)

    _is_configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the application hierarchy.

    The returned logger is a child of the root ``TelescopeApp`` logger,
    so it inherits all handlers configured by ``setup_logging()``.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.

    Example::

        from telescope_logger import get_logger
        logger = get_logger(__name__)
        logger.info("Module loaded")
    """
    if name == _root_logger_name or name == "main":
        return logging.getLogger(_root_logger_name)
    return logging.getLogger(f"{_root_logger_name}.{name}")


def set_gui_callback(callback: Callable[[str, str], None]) -> None:
    """Register the GUI log panel callback.

    After calling this, all log messages will also be forwarded to
    ``callback(message, tag)`` where *tag* is one of the GUI color tags
    (``"info"``, ``"success"``, ``"warning"``, ``"error"``, ``"cmd"``).

    Args:
        callback: Function accepting ``(message: str, tag: str)``.
    """
    global _gui_callback, _gui_handler

    _gui_callback = callback

    root = logging.getLogger(_root_logger_name)

    # Remove any previous GUI handler
    if _gui_handler is not None and _gui_handler in root.handlers:
        root.removeHandler(_gui_handler)

    # Create and attach the new handler (no formatter -- we send raw messages)
    _gui_handler = _GUICallbackHandler()
    _gui_handler.setLevel(logging.DEBUG)
    # Simple formatter: message only (the GUI adds its own timestamp)
    _gui_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(_gui_handler)


def remove_gui_callback() -> None:
    """Unregister the GUI log panel callback."""
    global _gui_callback, _gui_handler

    _gui_callback = None
    if _gui_handler is not None:
        root = logging.getLogger(_root_logger_name)
        if _gui_handler in root.handlers:
            root.removeHandler(_gui_handler)
        _gui_handler = None


def log_with_tag(logger: logging.Logger, level: int, message: str,
                 tag: str, *args) -> None:
    """Log a message with an explicit GUI tag override.

    This is useful when the GUI tag does not correspond to the standard
    level mapping (e.g., logging at INFO level but displaying with the
    ``"cmd"`` blue tag, or ``"usb"`` tag).

    Args:
        logger:  The logger to use.
        level:   Python logging level (e.g., ``logging.INFO``).
        message: Log message (may contain %-style format placeholders).
        tag:     GUI tag string (``"cmd"``, ``"usb"``, ``"rate"``, etc.).
        *args:   Format arguments for the message.
    """
    if logger.isEnabledFor(level):
        record = logger.makeRecord(
            logger.name, level, "(unknown)", 0, message, args, None
        )
        record.gui_tag = tag  # type: ignore[attr-defined]
        logger.handle(record)


def set_log_level(console_level: Optional[int] = None,
                  file_level: Optional[int] = None) -> None:
    """Change log levels at runtime.

    Allows the user to raise or lower verbosity without restarting the
    application (e.g., via a config change or API call).

    Args:
        console_level: New minimum level for console output (e.g., ``logging.WARNING``).
        file_level:    New minimum level for file output (e.g., ``logging.INFO``).
    """
    root = logging.getLogger(_root_logger_name)
    for handler in root.handlers:
        if isinstance(handler, MemoryHandler):
            # MemoryHandler wraps the RotatingFileHandler
            if file_level is not None:
                handler.setLevel(file_level)
                if handler.target:
                    handler.target.setLevel(file_level)
        elif isinstance(handler, RotatingFileHandler):
            if file_level is not None:
                handler.setLevel(file_level)
        elif isinstance(handler, logging.StreamHandler):
            if console_level is not None:
                handler.setLevel(console_level)


# ---------------------------------------------------------------------------
#  Legacy API (backward-compatible wrapper)
# ---------------------------------------------------------------------------
class TelescopeLogger:
    """Legacy wrapper for backward compatibility with existing code.

    Existing code that does::

        from telescope_logger import TelescopeLogger
        self.file_logger = TelescopeLogger(name="TelescopeApp", log_file="app.log")
        self.file_logger.info("hello")

    continues to work.  Internally it delegates to ``setup_logging()`` and
    ``get_logger()``.
    """

    def __init__(
        self,
        name: str = "TelescopeApp",
        log_file: str = "telescope_app.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        level: int = logging.DEBUG,
    ):
        setup_logging(
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        self.logger = get_logger(name)

    def debug(self, message: str):
        """Log at DEBUG level."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log at INFO level."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log at WARNING level."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log at ERROR level."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log at CRITICAL level."""
        self.logger.critical(message)

    def exception(self, message: str):
        """Log an exception with traceback."""
        self.logger.exception(message)

    def get_log_file_path(self) -> Optional[str]:
        """Return the path to the log file, or None."""
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                return handler.baseFilename
        # Check parent handlers
        parent = self.logger.parent
        if parent:
            for handler in parent.handlers:
                if isinstance(handler, RotatingFileHandler):
                    return handler.baseFilename
        return None
