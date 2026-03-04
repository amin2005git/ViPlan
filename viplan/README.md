### Main library

Structure and usage:
- `experiments/`: experiment scripts for the benchmark. The two main entry points are:
  - `benchmark_vlm_as_grounder.py` — VLM-as-Grounder experiments (works with any environment)
  - `benchmark_vlm_as_planner.py` — VLM-as-Planner experiments (works with any environment)
  - `benchmark_igibson_oracle.py` — Oracle planner baseline for iGibson
  - `vlm_grounder/` — helper modules for the VLM-as-Grounder pipeline (state management, checks, planning, VLM queries)
- `models/`: all model classes, most models are implemented with either transformers or vLLM. Extend them to add more models
- `planning/`: code for the blocksworld simulated environment and the iGibson client
- `rendering/`: code for the blocksworld renderer
- `code_helpers.py`: shared utilities (model loading, domain configuration, logging)

We recommend looking at `experiments/` to get started. See the [main README](../README.md) for usage examples and experiment variants.

Back to [Main Documentation](../README.md).
