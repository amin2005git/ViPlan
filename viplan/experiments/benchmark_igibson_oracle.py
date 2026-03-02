import os
import copy
import fire
import json
import random
from collections import deque

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from viplan.experiments.vlm_grounder.questions import get_preconditions_predicates, get_effects_predicates
from viplan.experiments.vlm_grounder.state import update_problem
from viplan.experiments.vlm_grounder.planning import get_plan

from viplan.planning.igibson_client_env import iGibsonClient
from viplan.code_helpers import get_logger, get_unique_id


goal_templates = {
    'reachable': {
        True:  "the {0} needs to be reachable by the agent",
        False: "the {0} needs to be unreachable by the agent"
    },
    'holding': {
        True:  "the agent needs to be holding the {0}",
        False: "the agent must not be holding the {0}"
    },
    'open': {
        True:  "the {0} needs to be open",
        False: "the {0} needs to be closed"
    },
    'ontop': {
        True:  "the {0} needs to be on top of the {1}",
        False: "the {0} must not be on top of the {1}"
    },
    'inside': {
        True:  "the {0} needs to be inside the {1}",
        False: "the {0} must not be inside the {1}"
    },
    'nextto': {
        True:  "the {0} needs to be next to the {1}",
        False: "the {0} must not be next to the {1}"
    }
}

def get_goal_str(env):
    goal_fluents = env.goal_fluents
    goal_string = ""

    for fluent in goal_fluents:
        value = True
        if fluent.is_not():
            value = False
            fluent = fluent.args[0]
        name = fluent.fluent().name
        args = fluent.args

        descr = goal_templates[str(name)][value].format(*[arg.object().name for arg in args])
        goal_string += descr + "\n"
    
    goal_string = goal_string.strip()
    return goal_string

def test_predicates(predicates_to_test, env):
    
    results = {}
    for predicate in predicates_to_test:
        key = list(predicate.keys())[0]
        pddl_expected_value = predicate[key]
        predicate = key.split(" ")[0]
        args = ",".join(key.split(" ")[1:])
        vlm_state_value = env.state[predicate][args] if predicate in env.state and args in env.state[predicate] else False # Assume False if not in state
        results[key] = (vlm_state_value == pddl_expected_value, vlm_state_value, pddl_expected_value)
    
    results["all_match"] = all([result[0] for result in results.values()])
    return results

def replan(env, logger):
    new_problem = update_problem(env.state, env.problem)
    plan_result = get_plan(new_problem, logger)
    if plan_result is None:
        logger.warning("Breaking out of episode due to error in the planner")
        return None
    elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        logger.warning("Breaking out of episode due to planner not finding a plan")
        return None
    else:
        logger.info("Replanning successful")
        return plan_result.plan

def test_action(action, env, logger):
    action_predictions = 0
    grounded_args = {param.name: str(value) for param, value in zip(action.action.parameters, action.actual_parameters)}
    preconditions = action.action.preconditions
    effects = action.action.effects
    previous_state = copy.deepcopy(env.state)
    
    precondition_preds = get_preconditions_predicates(env, preconditions, grounded_args)
    action_predictions += len(precondition_preds)
    logger.info(f"Precondition predicates: {precondition_preds}")
    precond_results = test_predicates(precondition_preds, env)
    if not precond_results["all_match"]:
        logger.warning(f"Precondition predicates do not match PDDL model:")
        for k, v in precond_results.items():
            if k != "all_match":
                if not v[0]:
                    logger.warning(f"Predicate {k} does not match: PDDL expected {v[2]}, true state is {v[1]}")
        return False, action_predictions
    
    legal, info = env.apply_action(action)
    if not legal:
        logger.warning(f"Action {action.action.name} with params {action.actual_parameters} is not legal: {info}")
        return False, action_predictions
    
    effects_preds = get_effects_predicates(env, effects, grounded_args, previous_state)
    action_predictions += len(effects_preds)
    logger.info(f"Effect predicates: {effects_preds}")
    effects_results = test_predicates(effects_preds, env)
    if not effects_results["all_match"]:
        logger.warning(f"Effect predicates do not match PDDL model: {effects_results}")
        return False, action_predictions
    return True, action_predictions

def test_plan(env, logger=None, max_steps=10):
    if logger is None:
        logger = get_logger("info")
    env.reset()
    
    goal_string = get_goal_str(env)
    logger.info(f"Goal: {goal_string}")
    plan_predictions = 0

    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts
    plan = get_plan(env.problem, logger)
    if plan is None:
        logger.warning("Initial plan not found")
        return False, 0, plan_predictions
    else:
        logger.info("Initial plan found")
        logger.info(f"Plan: {plan.plan}")
    action_queue = deque(plan.plan.actions)
    
    replans = 0
    while action_queue and max_steps > 0:
        logger.info(f"Steps left: {max_steps}")
        action = action_queue.popleft()
        max_steps -= 1
        
        logger.info(f"Testing action {action.action.name} with params {action.actual_parameters}")
        success, action_predictions = test_action(action, env, logger)
        plan_predictions += action_predictions
        logger.info(f"Action predictions: {action_predictions}")
        if not success:
            logger.warning(f"Action {action.action.name} with params {action.actual_parameters} failed")
            new_plan = replan(env, logger)
            replans += 1
            if new_plan is None:
                logger.warning("Replanning failed")
                return False, replans, plan_predictions
            else:
                logger.info("Replanning successful")
                logger.info(f"New plan: {new_plan}")
                action_queue = deque(new_plan.actions)
                continue
            
        logger.info(f"Action {action.action.name} with params {action.actual_parameters} succeeded")
        
        if env.goal_reached:
            logger.info("Goal reached")
            return True, replans, plan_predictions
    logger.info(f"Plan finished with {len(action_queue)} actions left")
    logger.info(f"Plan predictions: {plan_predictions}")
    logger.warning("Max steps reached without reaching goal")
    return False, replans, plan_predictions

def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    base_url: str,
    seed: int = 1,
    problem_id: int = None,
    output_dir: os.PathLike = None,
    log_level ='info',
    max_steps: int = 10,
    **kwargs):
    
    random.seed(seed)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
        
    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts

    results = {}
    metadata = os.path.join(problems_dir, "metadata.json")
    assert os.path.exists(metadata), f"Metadata file {metadata} not found"
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    problem_files = [f"{problems_dir}/{problem}" for problem in metadata.keys()]
    
    if problem_id is not None:
        assert problem_id < len(problem_files), f"Problem ID {problem_id} out of range"
        problem_files = [problem_files[problem_id]]
    
    for problem_file in problem_files:
        logger.info(f"Loading problem {problem_file}")
        
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        task = metadata[os.path.basename(problem_file)]['activity_name']
        scene_instance_pairs = metadata[os.path.basename(problem_file)]['scene_instance_pairs']
        logger.info(f"Found {len(scene_instance_pairs)} scene-instance pairs for task {task}")
        for scene_id, instance_id in scene_instance_pairs:
            env = iGibsonClient(task=task, scene_id=scene_id, instance_id=instance_id, problem=problem, base_url=base_url, logger=logger)

            # Run planning loop
            success, replans, plan_predictions = test_plan(env, logger=logger, max_steps=max_steps)

            # Store results
            problem_results = {'success':success, 'replans':replans, 'plan_predictions':plan_predictions}
            results[f"{problem_file}_{scene_id}_{instance_id}"] = problem_results
    
    # Compute some statistics
    total_tasks_completed = 0
    total_replans = 0
    total_predictions = 0
    for problem_file, problem_results in results.items():
        if problem_results is None:
            continue
        total_tasks_completed += 1 if problem_results['success'] else 0
        total_replans += problem_results['replans']
        total_predictions += problem_results['plan_predictions']
        
    task_completion_rate = total_tasks_completed / len(results) if len(results) > 0 else 0
    average_replans_per_task = total_replans / len(results) if len(results) > 0 else 0
    average_predictions_per_task = total_predictions / len(results) if len(results) > 0 else 0
    
    results['statistics'] = {
        'total_tasks_completed': total_tasks_completed,
        'task_completion_rate': task_completion_rate,
        'average_replans_per_task':average_replans_per_task,
        'average_predictions_per_task': average_predictions_per_task,
    }
    
    results['metadata'] = {
        'model': 'oracle_planner',
        'seed': seed,
        'max_steps': max_steps,
        'job_id': unique_id,
    }

    logger.info(f"Tasks completed: {total_tasks_completed}")
    logger.info(f"Task completion rate: {task_completion_rate}")
    logger.info(f"Average number of replans per task: {average_replans_per_task}")
    
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    logger.info(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)