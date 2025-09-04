## Running with Hydra

Hydra is configured via YAML files under `configs/` and does not change your working directory (`hydra.job.chdir=false`). Each run gets a unique folder at `runs/<timestamp>_<analysis>`; we write a resolved config JSON there for provenance.

- Install dependencies (add if missing): `pip install hydra-core omegaconf`

- Single run (uses defaults in `configs/config.yaml`):
  - `python scripts/run_experiment.py`

- Override parameters on the command line:
  - `python scripts/run_experiment.py analysis=checkerboard dataset=hcpa harmonics.n_modes=64`

- Multirun sweeps (Hydra `-m`):
  - `python scripts/run_experiment.py -m analysis=heatmaps,checkerboard harmonics.n_modes=64,128 dataset=camcan`
  - Sweep outputs go under `runs/multirun/<timestamp>/<analysis>/<num>`.

- Where things go:
  - Resolved config: `runs/.../config_resolved.json`
  - Your code should save outputs under `runs/.../` (script writes there explicitly).

Config tree highlights:
- `configs/config.yaml` composes defaults (dataset, analysis, harmonics, viz, hydra).
- `configs/hydra.yaml` pins behavior (no chdir, deterministic run/sweep dirs).
- Swap datasets/analyses via `configs/dataset/*.yaml` and `configs/analysis/*.yaml`.

