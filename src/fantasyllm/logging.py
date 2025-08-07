"""Structured logging configuration for Fantasy LLM."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def configure_logging() -> None:
    """Configure structured logging."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FILENAME,
                           structlog.processors.CallsiteParameter.LINENO]
            ),
            structlog.dev.ConsoleRenderer()
            if settings.log_format == "console"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Set up file logging
    file_handler = logging.FileHandler(log_dir / "fantasy_llm.log")
    file_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    if settings.log_format == "json":
        file_formatter = structlog.processors.JSONRenderer()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add structured logging to classes."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize logger for subclasses."""
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(cls.__name__)

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance."""
        return get_logger(self.__class__.__name__)


def log_model_info(model_name: str, parameters: Dict[str, Any]) -> None:
    """Log model information."""
    logger = get_logger("model")
    logger.info(
        "Model loaded",
        model_name=model_name,
        parameters=parameters,
    )


def log_training_step(
    step: int, loss: float, learning_rate: float, **kwargs: Any
) -> None:
    """Log training step information."""
    logger = get_logger("training")
    logger.info(
        "Training step",
        step=step,
        loss=loss,
        learning_rate=learning_rate,
        **kwargs,
    )


def log_inference(prompt: str, response: str, **kwargs: Any) -> None:
    """Log inference information."""
    logger = get_logger("inference")
    logger.info(
        "Inference completed",
        prompt_length=len(prompt),
        response_length=len(response),
        **kwargs,
    )


def log_error(error: Exception, context: Dict[str, Any]) -> None:
    """Log error with context."""
    logger = get_logger("error")
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        context=context,
        exc_info=True,
    )


# Initialize logging on import
configure_logging()