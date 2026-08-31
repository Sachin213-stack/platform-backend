import os
import sys
from alembic.config import Config
from alembic import command
from app.core.logging import logger


def run_upgrade(revision: str = "head") -> None:
    """Programmatically run Alembic migrations up to the specified revision."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ini_path = os.path.join(base_dir, "alembic.ini")

    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"Alembic configuration not found at {ini_path}")

    alembic_cfg = Config(ini_path)
    logger.info(f"Running database migrations to revision: {revision}")
    command.upgrade(alembic_cfg, revision)
    logger.info("Database migrations completed successfully.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "head"
    run_upgrade(target)
