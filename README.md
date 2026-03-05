# ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models

This codebase contains the implementation of the ViPlan benchmark.

![ViPlan](img/overview.png)

## Project structure

The project is divided into the following main sections:

- Source code: [viplan](viplan/README.md)
- Notebooks: [notebooks](notebooks/README.md) (mostly to visualize results)
- Scripts to run the benchmark: [sh_scripts](sh_scripts/README.md)
- Data: [data](data/README.md)

## Installation

The ViPlan benchmark is made up of several components, including the main experiment code and specific code for the two environments (Blocksworld and Household).

### Experiments

To run the experiments, you need to install the required packages. We recommend using mamba and provide an environment file for easy installation. The virtual environment requirements can be found at `environment.yml`, and it can be created as prefered. Here we report examples using `mamba`.
Using `mamba`:

```bash
mamba env create -p ./viplan_env -f environment.yml
mamba activate ./viplan_env
```

> [!WARNING]
> Using conda is not ufficially supported, but if you want, swap mamba with conda everywhere (also in the sh_scripts) and you should be good.
>  e.g.

```bash
conda env create -p ./viplan_env -f environment.yml
conda activate ./viplan_env
```

If you wish to use Flash Attention, it needs to be installed separately with the following command:

```bash
pip install flash-attn --no-build-isolation
```

> [!WARNING]
> At the time of writing, Molmo has [an issue](https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/44) with the latest version of `transformers` (>= 4.51.0). To run Molmo, please downgrade `transformers` to version 4.50.3 with `pip install transformers==4.50.3`.

### Environments

The Blocksworld environment is based on the [Photorealistic Blocksworld](https://github.com/IBM/photorealistic-blocksworld) renderer, which is based on Blender. To install the Blender-based renderer, from the root directory of the repository, run the following commands:

```bash
./setup_blocksworld.sh
```

Additionally, the libxi package needs to be installed (e.g., `sudo apt-get install libxi`) or available in the cluster.

#### iGibson

Here is the list of specific requirements to use iGibson:

- `apptainer` (former Singularity)
- Encription key to be requested at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform)

The Household environment is instead based on a custom version of [iGibson](https://github.com/StanfordVL/iGibson). 
To install the environment, first clone our fork of iGibson:

```bash
git clone --depth 1 --single-branch --branch release_viplan https://github.com/nicoladainese96/iGibson.git ./iGibson --recursive
git clone https://github.com/StanfordVL/behavior.git
```

Since iGibson requires specific packages, we recommend running it inside a container. Our code is designed to work with [Apptainer](https://apptainer.org). To pull the image, run:

```bash
apptainer cache clean
apptainer pull docker://igibson/igibson:latest
```
This will create a file called `igibson_latest.sif` (it should take approximately 15 minutes), which is expected to be in the root directory. This file is a Singularity image that contains all the dependencies needed to run iGibson. To open a shell inside the container run:
```bash
apptainer exec --nv igibson_latest.sif bash
```

Then, install the iGibson dependencies from inside the container:

```bash
python -m venv --system-site-packages ./igibson_env
source igibson_env/bin/activate
pip install -e ./iGibson
pip install -e ./behavior
pip install notebook pyquaternion shapely uvicorn fastapi unified_planning
pip install unified_planning[engines]
```

Afterwards, the iGibson custom assets need to be downloaded following the instructions at [this page](https://stanfordvl.github.io/iGibson/dataset.html):

To download the assets, run:

```bash
cd iGibson
wget --no-check-certificate https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz
mkdir igibson/data
tar -xzvf ig_dataset.tar.gz -C ./igibson/data
```

Then, still in the iGibson folder, from inside the container run:
```bash
python -m igibson.utils.assets_utils --download_assets
python -m igibson.utils.assets_utils --download_demo_data
```

As some of the assets are encrypted, you will need to download the key provided by the iGibson team. The key can be requested by filling out the form at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform) and then needs to be placed inside the `iGibson` folder under `igibson/data/igibson.key`.

After this, the iGibson environment is ready to be used. For the benchmark, we use a client-server architecture, where the server runs inside the container and the client runs in the main execution environment. Scripts are provided in the `sh_scripts` folder to run the server and the client.

> [!WARNING]
> iGibson will create many temporary files under `iGibson/igibson/data/ig_dataset/scene_instances`, which are not removed automatically. The folder is safe to delete to clear up space.

## Benchmark

### API keys

In order to run some open-source models, you might need to accept their conditions on the huggingface hub. Then, you can include your token in the bash environment by running the following command:

```bash
export HF_TOKEN=<your_token>
```

Similarly, in order to run closed-source models, include your API key in the bash environment by running the following command:

```bash
export OPENAI_API_KEY=<your_key>
export GEMINI_API_KEY=<your_key>
export ANTHROPIC_API_KEY=<your_key>
```

### Problem instances

The benchmark has three difficulty levels — **simple**, **medium**, and **hard** — each containing 25 problem instances. Each problem instance is a concrete task that the agent must solve starting from a specific initial scene.

**Total problem instances per domain:**

| Domain | Simple | Medium | Hard | **Total** |
|--------|--------|--------|------|-----------|
| ViPlan-BW (Blocksworld) | 25 | 25 | 25 | **75** |
| ViPlan-HH (Household/iGibson) | 25 | 25 | 25 | **75** |

**How problems are stored:**

- **ViPlan-BW:** Each of the 75 instances is a separate PDDL file (25 files per difficulty level).
- **ViPlan-HH:** The 75 instances are represented differently. There are only 16 PDDL files in total (5–6 per difficulty level), each defining a *task template* (the objects and goal). The 25 instances per difficulty level come from pairing each template with multiple `(scene_id, instance_id)` combinations listed in `metadata.json`. A `scene_id` (e.g. `"Ihlen_0_int"`) identifies a specific iGibson house layout; an `instance_id` (e.g. `0`, `20`) selects a particular object placement within that house. Each unique `(PDDL file, scene_id, instance_id)` triplet is one problem instance.

**Goals and the VLM:** Goals are defined symbolically in the PDDL files (e.g. `(ontop hardback_1 shelf_1)`). At runtime the experiment scripts translate those conditions into plain English using the `goal_templates` dictionary in `viplan/code_helpers.py`, and insert the result into the prompt at the `{goal_string}` placeholder before calling the VLM. The VLM **receives** the goal — it is told what to achieve — but it does not generate or infer the goal itself.

**How problems are created:** ViPlan-BW problems are generated programmatically using `viplan/planning/blocksworld_problem_generator.py` (random block sampling, random initial/goal layout, classical-planner validation for reachability and plan-length difficulty). ViPlan-HH problems use human-authored PDDL task templates from the iGibson Activity Benchmark, each paired with different house scenes and object placements.

See [data/README.md](data/README.md) for the full breakdown.

### Running experiments

The benchmark consists of two main experiment types, each implemented as an environment-agnostic Python script:

- **VLM-as-Grounder** (`viplan.experiments.benchmark_vlm_as_grounder`): The VLM predicts symbolic predicates from visual observations, which are then used by a classical planner to generate actions.
- **VLM-as-Planner** (`viplan.experiments.benchmark_vlm_as_planner`): The VLM directly outputs actions from visual observations.

Both scripts work with any supported environment by specifying the `--domain_name` parameter (`viplan-bw` for Blocksworld, `viplan-hh` for Household/iGibson).

#### VLM-as-Grounder

```bash
python3 -m viplan.experiments.benchmark_vlm_as_grounder \
  --model_name "OpenGVLab/InternVL3-8B" \
  --domain_name "viplan-bw" \
  --domain_file "data/planning/blocksworld/domain.pddl" \
  --problems_dir "data/planning/blocksworld/problems/simple" \
  --prompt_path "data/prompts/benchmark/blocksworld/prompt.md" \
  --root_path "." \
  --output_dir "results/my_experiment" \
  --seed 1
```

**Experiment variants for VLM-as-Grounder:**

| Variant | Flag(s) | Description |
|---------|------|-------------|
| Default | *(none)* | Standard Yes/No QA prompt (`prompt.md`) |
| Chain-of-Thought (CoT) | `--use_cot_prompt` (shell scripts) | Uses the CoT prompt variant (`prompt_cot.md`) |
| With memory (Mem) | `--include_prompt_history` | Injects previous-step failure context into the VLM prompt |
| Mem + CoT | `--include_prompt_history --use_cot_prompt` | Combines CoT prompting with memory |

#### VLM-as-Planner

```bash
python3 -m viplan.experiments.benchmark_vlm_as_planner \
  --model_name "OpenGVLab/InternVL3-8B" \
  --domain_name "viplan-bw" \
  --domain_file "data/planning/blocksworld/domain.pddl" \
  --problems_dir "data/planning/blocksworld/problems/simple" \
  --prompt_path "data/prompts/planning/vila_blocksworld_json.md" \
  --root_path "." \
  --output_dir "results/my_experiment" \
  --max_steps 10 \
  --seed 1
```

**Experiment variants for VLM-as-Planner:**

| Variant | Flag(s) | Prompt selected |
|---------|---------|-----------------|
| Default | *(none)* | `vila_{env}_json.md` |
| Chain-of-Thought (CoT) | `--use_cot_prompt` | `vila_{env}_json_cot.md` |
| Act | `--act_prompt` | `act_{env}_json.md` |
| Act + CoT | `--use_cot_prompt --act_prompt` | `react_{env}_json.md` |

where `{env}` is `blocksworld` or `igibson`. The prompt is selected automatically by the shell scripts; when running Python directly, pass the desired prompt via `--prompt_path`.

#### iGibson-specific notes

For iGibson experiments, replace `--root_path` with `--base_url` pointing to the running iGibson server:

```bash
python3 -m viplan.experiments.benchmark_vlm_as_grounder \
  --model_name "OpenGVLab/InternVL3-8B" \
  --domain_name "viplan-hh" \
  --domain_file "data/planning/igibson/domain.pddl" \
  --problems_dir "data/planning/igibson/simple" \
  --prompt_path "data/prompts/benchmark/igibson/prompt.md" \
  --base_url "http://localhost:8900" \
  --output_dir "results/my_experiment" \
  --seed 1
```

An oracle planner baseline is also available for iGibson:

```bash
python3 -m viplan.experiments.benchmark_igibson_oracle \
  --base_url "http://localhost:8900" \
  --domain_file "data/planning/igibson/domain.pddl" \
  --problems_dir "data/planning/igibson/simple" \
  --output_dir "results/oracle" \
  --max_steps 10 \
  --seed 1
```

> [!NOTE]
> The iGibson environment uses a client-server architecture. The simulation server must be started inside the Apptainer container before running experiments. See the [iGibson setup](#igibson) section and the scripts in `sh_scripts/` for details on starting the server.

#### Using the shell scripts

We also provide bash scripts to run experiments locally as well as SLURM scripts to run on a cluster. The scripts are located in the `sh_scripts` folder. See the [sh_scripts README](sh_scripts/README.md) for more details on available flags and how to use them. If you are using a different cluster manager, you may need to modify the SLURM scripts at `sh_scripts/slurm_cluster` accordingly.

## Results

We include all the results from the experiments reported in the paper in the `results` folder. To process and visualize them, we provide Jupyter notebooks in the `notebooks` folder. This reproduces exactly all the Figures and Tables reported in the paper.

## Extending ViPlan

ViPlan can be easily extended by the community to include new domains, models and methods.

### Adding new domains

In order to add a new domain, the following steps are needed:

- Add a new subfolder in `data/planning/` with a PDDL domain file and per-split problem files.
- Implement the domain simulator under `viplan/planning/`, as a subclass of `PlanningSimulator`.
- Update `get_domain_config` in `viplan/code_helpers.py` with the domain-specific logic.
- Add prompts under `data/prompts/`.

Once this is done, the `sh_scripts` can run the new domain with minimal changes.

### Adding new models

ViPlan provides integration with vLLM to run open-source VLMs. With updates to the framework, this should support new models as they are released, with minimal changes needed. 
We also provide interfaces with the OpenAI, Gemini and Anthropic APIs to run closed-source models, which should be able to support new models from these providers with minimal changes.

### Adding new methods

Edits to the existing VLM-as-planner and VLM-as-grounder methods can be made in the `viplan/experiments` folder. To add a completely new method, a new script can be added to the same folder, following the same structure as the existing ones.