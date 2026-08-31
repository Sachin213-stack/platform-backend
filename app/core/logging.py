import logging
import json
import sys
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Dict

# Context variables for request tracing
request_id_ctx: ContextVar[str] = ContextVar("request_id_ctx", default="")
business_id_ctx: ContextVar[str] = ContextVar("business_id_ctx", default="")


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured observability."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "business_id": business_id_ctx.get(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger("aicto")
