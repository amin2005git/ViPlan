import os
import copy
import fire
import json
import torch
import random
import transformers

from typing import Any

import unified_planning
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from viplan.code_helpers import get_logger, get_unique_id, get_domain_config, load_vlm
from viplan.experiments.vlm_grounder.planning import (
    check_plan,
    compute_metrics,
    get_plan,
    compute_enumeration_metrics,
)
from viplan.experiments.vlm_grounder.questions import get_questions
from viplan.experiments.vlm_grounder.state import update_problem, update_vlm_state
from viplan.experiments.vlm_grounder.vlm_query import get_enumeration_results

def main(
    problems_dir: os.PathLike,
    domain_name: str, # Used for the config
    domain_file: os.PathLike,
    model_name: str,
    prompt_path: os.PathLike,
    base_url: str = None, # Optional, only required for ViPlan-HH
    root_path: os.PathLike = None, # Optional, only required for ViPlan-BW
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level ='info',
    replan: bool = True, # Try to replan if an action fails
    fail_probability: float = 0.0, # Probability of action failure, only for ViPlan-BW
    enumerate_initial_state: bool = False, # Enumerate initial state predicates (instead of using oracle)
    enumerate_replan: bool = True, # Enumerate predicates before replanning if there is a failure
    enum_batch_size: int = 64, # Batch size for enumeration
    max_steps: int = 20, # Max number of steps to take in the environment
    include_prompt_history: bool = False, # If True, inject previous-step failure context into the next VLM prompt
    **kwargs):
    
    # Ensure deterministic behavior (in theory)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
        
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")
        
    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts

    # Load model
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}." if torch.cuda.is_available() else "Using CPU.")
    if torch.cuda.is_available() and ('A100' in torch.cuda.get_device_name() or 'H100' in torch.cuda.get_device_name() or 'H200' in torch.cuda.get_device_name()):
        use_flash_attn = True
    else:
        use_flash_attn = False
    logger.info(f"Use flash attention: {use_flash_attn}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = load_vlm(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=0, device=device, dtype=dtype, use_flash_attention=use_flash_attn, **kwargs)
    logger.info(f"Loaded model {model_name} on device {device} with dtype {dtype}")
    
    results = {}
    metadata = os.path.join(problems_dir, "metadata.json")
    assert os.path.exists(metadata), f"Metadata file {metadata} not found"
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    problem_files = [problem for problem in metadata.keys()]
    problem_files = [f"{problems_dir}/{problem}" for problem in problem_files]
    env_factory, predicate_questions, argument_aliases, _, _, problem_iterator = get_domain_config(domain_name)
    
    for problem_file, scene_id, instance_id in problem_iterator(problem_files, metadata):
        task = metadata[os.path.basename(problem_file)]['activity_name'] if 'activity_name' in metadata[os.path.basename(problem_file)] else None
        problem_file = problem_file if problem_file.endswith('.pddl') else problem_file + '.pddl'
        logger.info(f"Loading problem {problem_file}")
        
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        results_key = f"{problem_file}_{scene_id}_{instance_id}" if scene_id is not None and instance_id is not None else f"{problem_file}"

        results[results_key] = {}
        try:
            env = env_factory(task=task, scene_id=scene_id, instance_id=instance_id, problem=problem, base_url=base_url, logger=logger, root_path=root_path, fail_probability=fail_probability, **kwargs)
        except Exception as e:
            logger.error(f"Could not load problem {problem_file}: {e}")
            results[results_key] = {
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            }
            continue
        
        if enumerate_initial_state:
            assert env.supports_full_enumeration, "Environment does not support full enumeration of initial state predicates, likely because it is partially observable."
            try:
                _ = get_plan(problem, logger) # For some reason we need to plan with the problem before updating it or it breaks ???
                
                logger.info(f"Original state:\n{str(env)}")
                logger.info("Enumerating initial state predicates")
                predicates = env.state
                print(f"Predicates: {predicates}")
                questions = get_questions(predicates, predicate_questions, argument_aliases)
                enum_results = get_enumeration_results(env, model, questions, prompt_path, logger, batch_size=enum_batch_size)
                stats = compute_enumeration_metrics(enum_results)
                results[problem_file]['initial_state_enum'] = {'results': enum_results, 'statistics': stats}
                logger.info(f"Enumeration accuracy: {stats['accuracy']:.2f}, Yes accuracy: {stats['yes_accuracy']:.2f}, No accuracy: {stats['no_accuracy']:.2f}")
                
                vlm_state, changed = update_vlm_state(copy.deepcopy(env.state), enum_results)
                logger.info(f"VLM state changed: {changed}")
                problem = update_problem(vlm_state, problem)
                
            except Exception as e:
                logger.error(f"Could not enumerate initial state: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results[problem_file].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
                continue
        else:
            vlm_state = copy.deepcopy(env.state) # Oracle state
        
        plan_result = get_plan(problem, logger)
        if plan_result is None:
            logger.warning("Breaking out of episode due to error in the planner")
            results[results_key].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
            continue
        elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
            logger.warning("No plan found.")
            results[results_key].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
            continue
        plan = plan_result.plan

        all_correct, action_results, replans, action_queue, goal_reached = check_plan(
            env=env,
            plan=plan,
            vlm_state=vlm_state,
            model=model,
            base_prompt=prompt_path,
            logger=logger,
            predicate_questions=predicate_questions,
            argument_aliases=argument_aliases,
            replan=replan,
            enumerate_replan=enumerate_replan,
            enum_batch_size=enum_batch_size,
            max_actions=max_steps,
            include_prompt_history=include_prompt_history,
        )
        results[results_key].update({
            'all_correct': all_correct,
            'goal_reached': goal_reached,
            'action_results': action_results,
            'replans': replans,
            'remaining_actions': [str(a) for a in action_queue]
        })

        # break
    
    predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio = compute_metrics(results, logger)
    logger.info(f"Predicate accuracy: {predicate_accuracy:.2f}, Macro predicate accuracy: {macro_predicate_accuracy:.2f}, Action accuracy: {action_accuracy:.2f}, Task accuracy: {task_accuracy:.2f}")
    logger.info(f"Fail ratio: {fail_ratio:.2f}")
    results['problem_stats'] = problem_stats
    results['predicate_stats'] = predicate_stats
    results['predicate_accuracy'] = predicate_accuracy
    results['macro_predicate_accuracy'] = macro_predicate_accuracy
    results['action_accuracy'] = action_accuracy
    results['task_accuracy'] = task_accuracy
    results['fail_ratio'] = fail_ratio
    results['metadata'] = {
        'model_name': model_name,
        'prompt_path': prompt_path,
        'problems_dir': problems_dir,
        'seed': seed,
        'replan': replan,
        'fail_probability': fail_probability,
        'enumerate_initial_state': enumerate_initial_state,
        'job_id': unique_id,
    }
    
    if enumerate_initial_state:
        problem_keys = [k for k in results.keys() if isinstance(results[k], dict) and 'initial_state_enum' in results[k]]
        enumeration_accuracy = sum([results[problem]['initial_state_enum']['statistics']['accuracy'] for problem in problem_keys]) / len(problem_keys) if len(problem_keys) > 0 else None
        results['enumeration_accuracy'] = enumeration_accuracy
        if enumeration_accuracy is not None:
            logger.info(f"Enumeration average accuracy: {enumeration_accuracy:.2f}")
        else:
            logger.info("No problems were enumerated")
        
        predicate_enumeration_accuracy = {}
        for problem in problem_keys:
            predicate_enum_stats = results[problem]['initial_state_enum']['statistics']['predicates']
            for predicate in predicate_enum_stats:
                if predicate not in predicate_enumeration_accuracy:
                    predicate_enumeration_accuracy[predicate] = []
                predicate_enumeration_accuracy[predicate].append(predicate_enum_stats[predicate]['accuracy'])
                        
        for predicate in predicate_enumeration_accuracy:
            avg_accuracy = sum(predicate_enumeration_accuracy[predicate]) / len(predicate_enumeration_accuracy[predicate]) if len(predicate_enumeration_accuracy[predicate]) > 0 else None
            if avg_accuracy is not None:
                logger.info(f"Predicate {predicate} average enumeration accuracy: {avg_accuracy:.2f}")
            else:
                logger.info(f"Predicate {predicate} had no enumeration accuracy")
            
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    fire.Fire(main)
