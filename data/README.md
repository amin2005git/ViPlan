### Data

In the following folder, you can find:
- Blocksworld files used for rendering in `blocksworld_rendering/`
- Custom chat templates for models like DeepSeekVL in `chat_templates/`
- PDDL problems and domains for the ViPlan-BW and ViPlan-HH in `planning/`
- All the prompts used in `prompts/`

### ViPlan-HH Problem Structure

The Household (iGibson) benchmark uses a different structure from Blocksworld:

- **Blocksworld (ViPlan-BW):** Each split contains 25 independent PDDL files, one per problem instance.
- **Household (ViPlan-HH):** Each split contains a smaller set of PDDL *template* files (5–6 per split), each paired with multiple scene/instance combinations listed in a `metadata.json` file. Together, each split sums to **25 problem instances**.

The `metadata.json` in each split (`simple/`, `medium/`, `hard/`) maps every PDDL file to:
- `activity_name`: the iGibson activity identifier used by the simulator
- `natural_language_goal`: a human-readable description of the goal for that problem
- `scene_instance_pairs`: a list of `[scene_id, instance_id]` pairs that constitute the 25 problem instances for the split

**Summary of ViPlan-HH problems:**

| Split  | PDDL templates | Problem instances |
|--------|---------------|-------------------|
| Simple | 5             | 25                |
| Medium | 6             | 25                |
| Hard   | 5             | 25                |
| **Total** | **16**     | **75**            |
