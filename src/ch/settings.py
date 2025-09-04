from __future__ import annotations

from pathlib import Path

try:
    # Pydantic v1
    from pydantic import BaseSettings
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Pydantic is required. Add `pydantic>=1.10,<2` to dependencies."
    ) from exc


ENV_PREFIX = "CH_"


class Settings(BaseSettings):
    """Single-source project configuration.

    - Uses environment variables with prefix `CH_` (and optional `.env`).
    - No YAML files are required; keep everything centralized here.
    - Example overrides:
        CH_DATA_ROOT=/abs/data CH_LAP_TYPE=combinatorial
    """

    # Paths
    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    backups_dir: Path = Path("data/backups")

    # Data files
    mat_filename: str = "nhw2022-network-harmonics-data.mat"

    # Algorithm defaults
    lap_type: str = "normalized"  # or "combinatorial"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = ENV_PREFIX
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ---------- Convenience ----------
    @property
    def mat_path(self) -> Path:
        return Path(self.data_root) / self.mat_filename

    def ensure_dirs(self) -> None:
        for p in [self.data_root, self.raw_dir, self.processed_dir, self.backups_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Return Settings instance (env > defaults)."""
    return Settings()


# Example usage:
#   from ch.settings import load_settings
#   s = load_settings()
#   s.ensure_dirs()
#   print(s.mat_path)
