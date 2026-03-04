# Scripts to run the benchmark

The ViPlan benchmark is designed to be run on SLURM clusters, and the scripts in this directory are tailored for that purpose. If you are using a different cluster manager, you may need to modify the scripts accordingly, or directly run the Python scripts in the `viplan/experiments` directory (see the [main README](../README.md) for direct Python usage).

> [!IMPORTANT]
> All sh_scripts are designed to be run from the root directory of the repository. (e.g. `cd ViPlan && ./sh_scripts/slurm_cluster/run_blocksworld.sh`)

## Structure

- `local/` — scripts for running experiments locally (sequential execution)
- `slurm_cluster/` — SLURM array job scripts for cluster execution

Within each, the "big" scripts are designed to run bigger VLMs that require two GPUs and the "cpu" scripts are designed to run API models that don't require GPUs (although a GPU is still requested for the renderer).

## Entry points

The two main entry points are `run_blocksworld.sh` and `run_igibson.sh` (located at `sh_scripts/slurm_cluster` to run on SLURM clusters; at `sh_scripts/local` to run locally), which are designed to run the Blocksworld and Household environments.

**Parameters for `run_blocksworld.sh` / `run_igibson.sh`:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--experiment_name` | string | Name used to organize result files |
| `--run_predicates` | boolean | Run VLM-as-Grounder experiments (default: `true` for BW, `false` for iGibson) |
| `--run_vila` | boolean | Run VLM-as-Planner experiments (default: `true`) |
| `--run_closed_source` | flag | Also run closed-source model variants |

Any additional arguments are forwarded to the individual experiment scripts.

## Experiment variants

The individual scripts under `scripts/` accept additional flags to select experiment variants:

**VLM-as-Grounder** (`benchmark_{env}_planning_array.sh`):

| Variant | Flag(s) | Description |
|---------|------|-------------|
| Default | *(none)* | Standard Yes/No QA prompt (`prompt.md`) |
| Chain-of-Thought (CoT) | `--use_cot_prompt` | Uses the CoT prompt variant (`prompt_cot.md`) |
| With memory (Mem) | `--include_prompt_history` | Injects previous-step failure context into the VLM prompt |
| Mem + CoT | `--include_prompt_history --use_cot_prompt` | Combines CoT prompting with memory |

**VLM-as-Planner** (`benchmark_{env}_vila_array.sh`):

| Variant | Flag(s) | Prompt selected |
|---------|---------|-----------------|
| Default | *(none)* | `vila_{env}_json.md` |
| Chain-of-Thought (CoT) | `--use_cot_prompt` | `vila_{env}_json_cot.md` |
| Act | `--act_prompt` | `act_{env}_json.md` |
| Act + CoT | `--use_cot_prompt --act_prompt` | `react_{env}_json.md` |

**Examples:**

```bash
# Run VLM-as-Grounder with Mem + CoT prompting
./sh_scripts/local/run_blocksworld.sh --run_vila false --use_cot_prompt --include_prompt_history --experiment_name cot_mem

# Run VLM-as-Planner with Act + CoT prompting
./sh_scripts/local/run_blocksworld.sh --run_predicates false --use_cot_prompt --act_prompt --experiment_name act_cot
```

Check the individual scripts for more details.

Back to [Main Documentation](../README.md).