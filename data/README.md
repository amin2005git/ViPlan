### Data

In the following folder, you can find:
- Blocksworld files used for rendering in `blocksworld_rendering/`
- Custom chat templates for models like DeepSeekVL in `chat_templates/`
- PDDL problems and domains for the ViPlan-BW and ViPlan-HH in `planning/`
- All the prompts used in `prompts/`

### ViPlan-HH Problem Structure

The benchmark has three difficulty levels: **simple**, **medium**, and **hard**. Each difficulty level contains 25 problem instances. A *problem instance* is a concrete task: a specific set of objects, their initial positions, and a goal state that the agent must achieve.

The two domains store their problems differently:

- **ViPlan-BW (Blocksworld):** Every instance is a separate PDDL file — 25 files per difficulty level, 75 files in total.
- **ViPlan-HH (Household/iGibson):** There are only 16 PDDL files in total (5–6 per difficulty level). Each file defines a *task template* — the object types and goal conditions. The 25 instances per difficulty level come from running each template in multiple different houses and object placements, recorded as `(scene_id, instance_id)` pairs in `metadata.json`.

**What `scene_id` and `instance_id` mean:**

iGibson includes multiple photorealistic house scenes. A `scene_id` (e.g. `"Ihlen_0_int"`, `"Beechwood_0_int"`) identifies a specific house layout. Within a house, objects can be placed in different random configurations; an `instance_id` (e.g. `0`, `20`, `21`) selects one such configuration. Each unique `(PDDL file, scene_id, instance_id)` triplet is one benchmark problem instance.

**What `metadata.json` contains:**

Each `metadata.json` maps a PDDL filename to:
- `activity_name`: the iGibson activity name used to load the task in the simulator
- `scene_instance_pairs`: list of `[scene_id, instance_id]` pairs — one per problem instance for that template

**How the goal is given to the VLM:**

Goals are defined symbolically in the PDDL files (e.g. `(ontop hardback_1 shelf_1)`). At runtime the experiment scripts translate those PDDL goal conditions into plain English using the `goal_templates` dictionary in `viplan/code_helpers.py`, and insert that text into the prompt at the `{goal_string}` placeholder before calling the VLM. So the VLM **receives** the goal as part of its prompt — it is told what the goal is, and its task is to plan or execute actions to achieve it. The VLM never generates or infers the goal on its own.

**How problems are created:**

- **ViPlan-BW (Blocksworld):** Problems are generated programmatically using `viplan/planning/blocksworld_problem_generator.py`. The generator randomly samples a set of colored blocks, creates a random initial layout (blocks distributed across columns), then independently generates a random goal layout. It validates each candidate problem with a classical planner to ensure the goal is reachable and that the plan length falls within difficulty-appropriate bounds (controlling simple/medium/hard). Duplicate problems are discarded and replaced. The resulting PDDL files are stored directly in `data/planning/blocksworld/problems/{simple,medium,hard}/`.
- **ViPlan-HH (Household/iGibson):** Problems are not generated programmatically. The PDDL task templates are taken from the iGibson Activity Benchmark — a set of real household activities (e.g. sorting books, cleaning drawers) whose goals and object types were defined by humans. Each template is paired with different iGibson house scenes (`scene_id`) and object placement seeds (`instance_id`) to form the 25 instances per difficulty level.

**Summary of ViPlan-HH problems:**

| Difficulty | PDDL templates | Problem instances |
|------------|---------------|-------------------|
| Simple     | 5             | 25                |
| Medium     | 6             | 25                |
| Hard       | 5             | 25                |
| **Total**  | **16**        | **75**            |

