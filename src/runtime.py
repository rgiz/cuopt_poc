from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


class _BenignCuOptLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("cuopt_sh_client"):
            return True
        message = record.getMessage().lower()
        if "log not found for request" in message:
            return False
        if "delete /cuopt/log" in message and "404" in message:
            return False
        return True


def configure_logging(name: str, level: str | None = None) -> logging.Logger:
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    if os.getenv("SUPPRESS_BENIGN_CUOPT_LOGS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        has_filter = any(isinstance(existing, _BenignCuOptLogFilter) for existing in root_logger.filters)
        if not has_filter:
            root_logger.addFilter(_BenignCuOptLogFilter())
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def run_step(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    if logger:
        logger.info("Running step: %s", " ".join(command))
    subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        check=True,
    )
