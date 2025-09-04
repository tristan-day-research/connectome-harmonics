import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    # Keep CWD stable; write into Hydra's run dir explicitly
    run_dir = Path(str(cfg.hydra.run.dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for provenance
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2))

    # Tiny summary
    ds = cfg.dataset
    print(
        f"[run] dataset={ds.name}, atlas={ds.atlas}, n_nodes={ds.n_nodes}, "
        f"analysis={cfg.analysis.name}, n_modes={cfg.harmonics.n_modes}"
    )

    # TODO: integrate your actual pipeline
    # Example (optional): hydrate existing Settings from Hydra cfg
    # from ch.settings import load_settings, Settings
    # s = load_settings()
    # s = Settings(
    #     data_root=Path(cfg.paths.data_root),
    #     processed_dir=Path(cfg.paths.processed_dir),
    #     log_level=cfg.logging.level,
    #     lap_type=cfg.harmonics.laplacian,
    # )
    # s.ensure_dirs()
    # from ch import data_handling as io
    # from ch import analysis, viz
    # results = analysis.run(...)
    # viz.save_figures(results, out_dir=run_dir / "figures")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
