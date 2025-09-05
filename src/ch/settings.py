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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Single-source project configuration.

    - Uses environment variables with prefix `CH_` (and optional `.env`).
    - No YAML files are required; keep everything centralized here.
    - Example overrides:
        CH_DATA_ROOT=/abs/data CH_LAP_TYPE=combinatorial
    """

    # Paths (default relative to repository root, not CWD)
    data_root: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_root / "raw"
    processed_dir: Path = data_root / "processed"
    backups_dir: Path = data_root / "backups"
    metadata_dir: Path = data_root / "metadata"

    # Object paths
    camcan_raw: Path = raw_dir / "raw_data_nhw2022-network-harmonics-data.mat"
    metadata_parquet: Path = metadata_dir / "subject_metadata.parquet"
    connectivity_parquet: Path = processed_dir / "connectivity_matrices.parquet"
    harmonics_parquet: Path = processed_dir / "connectome_harmonics.parquet"

    # Algorithm defaults
    lap_type: str = "normalized"  # or "combinatorial"

    # Logging
    log_level: str = "INFO"


    

    class Config:
        env_prefix = ENV_PREFIX
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self) -> None:
        for p in [self.data_root, self.raw_dir, self.processed_dir, self.backups_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)
    
    def configure_logging(self) -> None:
        """Configure logging for the application."""
        import logging
        
        # Only configure if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, self.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S',
                force=True  # Override any existing configuration
            )

    def __str__(self):
        return f"Settings(data_root={self.data_root}, raw_dir={self.raw_dir}, processed_dir={self.processed_dir})"


def load_settings() -> Settings:
    """Return Settings instance (env > defaults) with logging configured."""
    settings = Settings()
    settings.configure_logging()
    return settings


def ensure_logging() -> None:
    """Ensure logging is configured. Useful for scripts that don't use load_settings()."""
    import logging
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            force=True
        )


# Example usage:
#   from ch.settings import load_settings
#   s = load_settings()
#   s.ensure_dirs()
#   print(s.mat_path)
