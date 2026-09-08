import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variables for distributed request & tenant tracing
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id_ctx", default="")
business_id_ctx: ContextVar[str] = ContextVar("business_id_ctx", default="")

# Backwards compatibility alias for existing code referencing request_id_ctx
request_id_ctx: ContextVar[str] = correlation_id_ctx


def get_correlation_id() -> str:
    """Retrieve the current request/job correlation ID."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current request/job correlation ID."""
    correlation_id_ctx.set(correlation_id)


def get_business_id() -> str:
    """Retrieve the current scoped tenant/business ID."""
    return business_id_ctx.get()


def set_business_id(business_id: str) -> None:
    """Set the current scoped tenant/business ID."""
    business_id_ctx.set(business_id)


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter.
    Emits one valid JSON object per line containing core metadata,
    tracing contextvars (correlation_id, business_id), caller information,
    and formatted exception stack traces.
    """

    STANDARD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        corr_id = correlation_id_ctx.get()
        biz_id = business_id_ctx.get()

        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
            "correlation_id": corr_id,
            "request_id": corr_id,  # Included as standard alias for log aggregators
            "business_id": biz_id,
            "module": record.module,
            "line": record.lineno,
        }

        # Include any custom extra fields passed into the logger call
        for key, val in record.__dict__.items():
            if key not in self.STANDARD_ATTRS and not key.startswith("_"):
                try:
                    # Test JSON serializability; fallback to str
                    json.dumps(val)
                    log_entry[key] = val
                except (TypeError, OverflowError):
                    log_entry[key] = str(val)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """
    Clean, human-readable console formatter for local developer usability.
    Formats logs with timestamp, colored severity levels, module name,
    correlation/tenant tags, and indented stack traces.
    """

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        corr_id = correlation_id_ctx.get()
        biz_id = business_id_ctx.get()

        meta_parts = []
        if corr_id:
            meta_parts.append(f"cid:{corr_id[:8]}")
        if biz_id:
            meta_parts.append(f"biz:{biz_id[:8]}")
        meta_str = f" [{ ' | '.join(meta_parts) }]" if meta_parts else ""

        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:<7}{self.RESET}"
        else:
            level_str = f"{level:<7}"

        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)

        formatted = f"[{timestamp}] [{level_str}] [{record.name}]{meta_str} {msg}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


_logging_initialized = False


def setup_logging() -> None:
    """
    Configures centralized logging for AI-CTO backend:
    - RotatingFileHandler: Persistent structured JSON logging (default 10MB x 5 backups).
    - StreamHandler: stdout handler (JSON in production / Render, clean console in development).
    - Both handlers run simultaneously.
    - Suppresses third-party noise.
    - Wires Sentry SDK logging integration if configured.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    from app.core.config import settings

    # Determine log level (env LOG_LEVEL overrides, defaulting to DEBUG in dev, INFO in prod)
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        log_level = getattr(logging, env_level)
    elif settings.ENVIRONMENT == "development":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # 1. Console / Stdout Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Determine console formatting
    log_format = (settings.LOG_FORMAT or "auto").lower()
    if log_format == "json" or (log_format == "auto" and settings.ENVIRONMENT != "development"):
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())

    # 2. Rotating File Handler (Persistent JSON logs)
    log_file_path = settings.LOG_FILE_PATH or "logs/aicto.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())

    # 3. Configure Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [console_handler, file_handler]

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

    # 4. Sentry Integration (if configured)
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,        # Capture INFO and above as breadcrumbs
                event_level=logging.ERROR  # Send ERROR and above as Sentry alert events
            )
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                integrations=[sentry_logging],
            )
            root_logger.info("Sentry SDK initialized with logging integration")
        except ImportError:
            root_logger.warning("SENTRY_DSN configured but sentry-sdk package is not installed")
        except Exception as e:
            root_logger.warning("Failed to initialize Sentry SDK: %s", e, exc_info=True)

    _logging_initialized = True


# Named logger for application modules
logger = logging.getLogger("aicto")
