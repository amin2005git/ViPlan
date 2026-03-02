import copy
import os
import tempfile

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import unified_planning
from mloggers import Logger
from unified_planning.shortcuts import OneshotPlanner

from viplan.experiments.vlm_grounder.checks import check_action
from viplan.experiments.vlm_grounder.questions import get_questions
from viplan.experiments.vlm_grounder.state import update_problem, update_vlm_state
from viplan.experiments.vlm_grounder.vlm_query import (
    EnumResults,
    get_enumeration_results,
)
from viplan.planning.planning_simulator import PlanningSimulator


def get_plan(problem: unified_planning.model.Problem, logger: Logger) -> Optional[unified_planning.engines.PlanGenerationResult]:
    result = None
    try:
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            # This is needed to avoid temporary file conflicts created in cwd from the planner in job arrays
            with tempfile.TemporaryDirectory() as td:
                old_cwd = os.getcwd()
                try:
                    os.chdir(td)  # Change to temporary directory
                    result = planner.solve(problem)
                finally:
                    os.chdir(old_cwd)  # Restore original directory

            if result.status == unified_planning.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.debug("Fast Downward returned: %s" % result.plan)
            else:
                logger.warning("No plan found.")
    except Exception as e:
        logger.warning(f"Planner crashed with error: {e}")

    return result


def check_plan(
    env: PlanningSimulator,
    plan: unified_planning.plans.Plan,
    vlm_state: Dict[str, Dict[str, bool]],  # Initial state as perceived by the VLM
    model: Any,
    base_prompt: os.PathLike,
    logger: Logger,
    predicate_questions: Dict[str, str],
    argument_aliases: Dict[str, str],
    replan: bool = False,
    max_actions: int = 20,
    enumerate_replan: bool = False,
    enum_batch_size: int = 64,
    include_prompt_history: bool = False,
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], deque, bool]:
    """
    Checks a plan by executing its actions in the environment and verifying preconditions and effects using the VLM.
    Args:
        env (PlanningSimulator): The planning environment.
        plan (unified_planning.plans.Plan): The plan to check.
        vlm_state (Dict[str, Dict[str, bool]]): The initial state as perceived by the VLM.
        model (Any): The VLM model used for generating answers.
        base_prompt (os.PathLike): Path to the base prompt file.
        logger (Logger): Logger for debug information.
        predicate_questions (Dict[str, str]): Mapping of predicates to a question template.
        argument_aliases (Dict[str, str]): Mapping of argument names to aliases for grounding.
        replan (bool, optional): If True, replans if an action fails. Defaults to False.
        max_actions (int, optional): Maximum number of actions to execute before stopping. Defaults to 20.
        enumerate_replan (bool, optional): If True, enumerates the predicates in the new state after replanning. Defaults to False.
        enum_batch_size (int, optional): Batch size for enumeration queries. Defaults to 64.
    Returns:
        Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], deque, bool]:
            - all_correct: Whether all actions were executed correctly.
            - results: List of results for each action, including correctness and precondition/effect results.
            - replans: List of replans if any actions failed, including enumeration results if applicable.
            - action_queue: Remaining actions in the plan after execution (if any).
            - goal_reached: Whether the goal was reached after executing the plan.
    """

    all_correct = True
    results = []
    replans = []
    action_queue = deque(plan.actions)
    most_recent_action = None
    pending_prompt_history: Optional[str] = None
    while action_queue and len(results) < max_actions:
        action = action_queue.popleft()

        logger.info(f"Applying action {action}")

        # Debug: save env image
        # img_path = os.path.join("debug", f"env_before_{len(results)}.png")
        # os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # img = env.render()
        # img.save(img_path)

        try:
            # Only use history if the current action matches the action stored in the history
            history_to_use = None
            if pending_prompt_history is not None and str(action) in pending_prompt_history:
                history_to_use = pending_prompt_history
                print(f"Using pending prompt history for action {action}:\n{history_to_use}")
            elif pending_prompt_history is not None:
                print(
                    f"Not using pending prompt history for action {action} as it does not match the action.\nPending history:\n{pending_prompt_history}"
                )

            action_correct, preconditions_results, non_visible_precond_results, effects_results, action_state_correct, action_info, history_for_next = check_action(
                env,
                action,
                vlm_state,
                model,
                base_prompt,
                logger,
                predicate_questions,
                argument_aliases,
                prompt_history=history_to_use,
                include_prompt_history=include_prompt_history,
            )
            pending_prompt_history = None
            print(f"History for next:\n{history_for_next}")
            if include_prompt_history and history_for_next is not None and str(history_for_next).strip() != "":
                pending_prompt_history = history_for_next
        except Exception as e:
            logger.warning(f"Error while checking action {action}: {e}")
            import traceback

            traceback.print_exc()

            action_correct = False
            preconditions_results = {}
            non_visible_precond_results = {}
            effects_results = None
            action_state_correct = False
            action_info = None
            pending_prompt_history = None
            break

        results.append(
            {
                "action": str(action),
                "action_correct": action_correct,
                "action_state_correct": action_state_correct,
                "preconditions_results": preconditions_results,
                "non_visible_precond_results": non_visible_precond_results,
                "effects_results": effects_results,
                "action_info": action_info,
            }
        )
        if not action_correct:
            if "all_correct" in preconditions_results and not preconditions_results["all_correct"]:
                reason = "Preconditions not satisfied"
                failed_results = preconditions_results
            elif effects_results is None:
                reason = "Action was not legal"
                failed_results = {}
            elif not effects_results["all_correct"]:
                reason = "Not all effects were observed as expected"
                failed_results = effects_results
            else:
                reason = "Unknown"
            logger.warning(f"Action {action} failed: {reason}")
            if reason == "Action was not legal" and str(action) == str(most_recent_action):
                logger.warning(
                    "Action was not legal, but it was the same as the most recent action. Stopping as we're likely in a loop."
                )
                break
            try:
                all_correct = False

                if replan:
                    replans.append({})
                    logger.info("Replanning from newly observed state")

                    if enumerate_replan:
                        # Enumerate the predicates in the new state
                        logger.info(f"Enumerating visible predicates: {env.visible_predicates}")
                        questions = get_questions(
                            env.visible_predicates, predicate_questions, argument_aliases
                        )  # For partial observability, use the visible predicates, while the rest stays unchanged
                        enum_results = get_enumeration_results(
                            env, model, questions, base_prompt, logger, batch_size=enum_batch_size, prompt_history=pending_prompt_history
                        )
                        enum_metrics = compute_enumeration_metrics(enum_results)
                        replans[-1]["enum_results"] = enum_results
                        replans[-1]["enum_metrics"] = enum_metrics

                        vlm_state, changed = update_vlm_state(copy.deepcopy(env.state), enum_results)
                        # No need to update vlm state if not enumerating, as the effect results are already updated in check_action
                else:
                    break
            except Exception as e:
                logger.warning(f"Error while updating VLM state: {e}")
                import traceback

                traceback.print_exc()
                all_correct = False
                break

            new_problem = update_problem(vlm_state, env.problem)
            plan_result = get_plan(new_problem, logger)
            if plan_result is None:
                logger.warning("Breaking out of episode due to error in the planner")
                break  # Exit episode
            elif plan_result.status != unified_planning.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.warning("No plan found after replanning")
                break
            else:
                logger.info("Replan found")
                new_plan = plan_result.plan
                action_queue = deque(new_plan.actions)
                replans[-1].update({"step": len(results), "actions": [str(a) for a in new_plan.actions]})

        if len(action_queue) == 0:
            logger.info("All actions completed")
            break
        if len(results) >= max_actions:
            logger.warning("Max actions reached")
            break

        most_recent_action = copy.deepcopy(action)

    goal_reached = env.goal_reached
    logger.info(f"Goal reached: {goal_reached}")

    if all_correct and not goal_reached:
        logger.warning(f"All actions executed correctly, but goal not reached")

    return all_correct, results, replans, action_queue, goal_reached


def compute_enumeration_metrics(results: EnumResults) -> Dict[str, Any]:
    enum_results = {
        "accuracy": 0,
        "yes_accuracy": 0,
        "yes_correct": 0,
        "yes_total": 0,
        "no_accuracy": 0,
        "no_correct": 0,
        "no_total": 0,
        "predicates": {},
    }
    # For enumeration, there is no expected value from the PDDL model (as we're testing everything), so the accuracy is already based on the environment
    for question, (answer, expected_answer) in results.items():
        if answer == expected_answer:
            enum_results["accuracy"] += 1
            if expected_answer == "yes":
                enum_results["yes_correct"] += 1
            else:
                enum_results["no_correct"] += 1

        if expected_answer == "yes":
            enum_results["yes_total"] += 1
        else:
            enum_results["no_total"] += 1

        predicate = question.split(" ")[0]
        if predicate not in enum_results["predicates"]:
            enum_results["predicates"][predicate] = {
                "accuracy": 0,
                "yes_accuracy": 0,
                "yes_correct": 0,
                "yes_total": 0,
                "no_accuracy": 0,
                "no_correct": 0,
                "no_total": 0,
            }
        if answer == expected_answer:
            enum_results["predicates"][predicate]["accuracy"] += 1
            if expected_answer == "yes":
                enum_results["predicates"][predicate]["yes_correct"] += 1
            else:
                enum_results["predicates"][predicate]["no_correct"] += 1

        if expected_answer == "yes":
            enum_results["predicates"][predicate]["yes_total"] += 1
        else:
            enum_results["predicates"][predicate]["no_total"] += 1

    enum_results["accuracy"] /= len(results) if len(results) > 0 else None
    enum_results["yes_accuracy"] = (
        enum_results["yes_correct"] / enum_results["yes_total"] if enum_results["yes_total"] > 0 else None
    )
    enum_results["no_accuracy"] = (
        enum_results["no_correct"] / enum_results["no_total"] if enum_results["no_total"] > 0 else None
    )

    for predicate in enum_results["predicates"]:
        enum_results["predicates"][predicate]["accuracy"] /= (
            enum_results["predicates"][predicate]["yes_total"] + enum_results["predicates"][predicate]["no_total"]
        )
        enum_results["predicates"][predicate]["yes_accuracy"] = (
            enum_results["predicates"][predicate]["yes_correct"] / enum_results["predicates"][predicate]["yes_total"]
            if enum_results["predicates"][predicate]["yes_total"] > 0
            else None
        )
        enum_results["predicates"][predicate]["no_accuracy"] = (
            enum_results["predicates"][predicate]["no_correct"] / enum_results["predicates"][predicate]["no_total"]
            if enum_results["predicates"][predicate]["no_total"] > 0
            else None
        )

    return enum_results


def compute_metrics(results: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
    # task accuracy = task was fully completed,
    # action_accuracy = fraction of individual actions that were correctly predicted (in full)
    # predicate_accuracy = fraction of predicates that were correctly predicted
    # macro_predicate_accuracy = fraction of predicates that were correctly predicted, equally weighted independently of the number of predicates
    # fail_ratio = fraction of problems that never had a plan (wrong initial state)

    task_accuracy = (
        sum([results[problem]["goal_reached"] for problem in results if "goal_reached" in results[problem]])
        / len(results)
    )
    problem_stats = {}
    predicate_stats = {}

    # Problem stats for action accuracy
    for problem in results:
        problem_stats[problem] = {}
        try:
            # Action accuracy is computed on the actual state correctness (e.g. did the VLM correctly answer as to what it was seeing)
            problem_stats[problem]["action_correct"] = sum(
                [
                    action["action_state_correct"]
                    for action in results[problem]["action_results"]
                    if "action_state_correct" in action
                ]
            )
            problem_stats[problem]["action_total"] = len(
                [action for action in results[problem]["action_results"] if "action_state_correct" in action]
            )
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy}")
            problem_stats[problem]["action_total"] += len(results[problem]["remaining_actions"])
            problem_stats[problem]["remaining_actions"] = results[problem]["remaining_actions"]
            problem_stats[problem]["action_accuracy"] = (
                problem_stats[problem]["action_correct"] / problem_stats[problem]["action_total"]
                if problem_stats[problem]["action_total"] > 0
                else 0
            )
            problem_stats[problem]["failed"] = False
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy} (after remaining actions {len(results[problem]['remaining_actions'])})")
        except Exception as e:
            problem_stats[problem]["action_correct"] = 0
            problem_stats[problem]["action_total"] = 1  # count this as a first action that failed to normalize the metric
            problem_stats[problem]["failed"] = True
            print(f"Problem {problem} had no actions, likely never started due to wrong initial state")
            continue
    logger.debug(f"Problem stats: {problem_stats}")

    # Predicate stats for predicate accuracy
    for problem in results:
        # First add the enumeration results
        if "initial_state_enum" in results[problem]:
            logger.debug(f"Adding initial state enumeration results for problem {problem}")
            for predicate in results[problem]["initial_state_enum"]["statistics"]["predicates"]:
                if predicate not in predicate_stats:
                    predicate_stats[predicate] = {
                        "accuracy": 0,
                        "yes_accuracy": 0,
                        "yes_correct": 0,
                        "yes_total": 0,
                        "no_accuracy": 0,
                        "no_correct": 0,
                        "no_total": 0,
                    }
                predicate_stats[predicate]["yes_correct"] += results[problem]["initial_state_enum"]["statistics"][
                    "predicates"
                ][predicate]["yes_correct"]
                predicate_stats[predicate]["yes_total"] += results[problem]["initial_state_enum"]["statistics"][
                    "predicates"
                ][predicate]["yes_total"]
                predicate_stats[predicate]["no_correct"] += results[problem]["initial_state_enum"]["statistics"][
                    "predicates"
                ][predicate]["no_correct"]
                predicate_stats[predicate]["no_total"] += results[problem]["initial_state_enum"]["statistics"][
                    "predicates"
                ][predicate]["no_total"]
        else:
            logger.info(f"No initial state enumeration results for problem {problem}")

        # Check if there is enumeration in the replans
        if "replans" in results[problem] and results[problem]["replans"] is not None:
            logger.debug(f"Adding replan enumeration results for problem {problem}")
            for replan in results[problem]["replans"]:
                if "enum_metrics" in replan:
                    for predicate in replan["enum_metrics"]["predicates"]:
                        if predicate not in predicate_stats:
                            predicate_stats[predicate] = {
                                "accuracy": 0,
                                "yes_accuracy": 0,
                                "yes_correct": 0,
                                "yes_total": 0,
                                "no_accuracy": 0,
                                "no_correct": 0,
                                "no_total": 0,
                            }
                        predicate_stats[predicate]["yes_correct"] += replan["enum_metrics"]["predicates"][predicate][
                            "yes_correct"
                        ]
                        predicate_stats[predicate]["yes_total"] += replan["enum_metrics"]["predicates"][predicate][
                            "yes_total"
                        ]
                        predicate_stats[predicate]["no_correct"] += replan["enum_metrics"]["predicates"][predicate][
                            "no_correct"
                        ]
                        predicate_stats[predicate]["no_total"] += replan["enum_metrics"]["predicates"][predicate][
                            "no_total"
                        ]
        else:
            logger.info(f"No replan enumeration results for problem {problem}")

            def update_predicate_stats(predicate_stats, results):
                for key, res in results.items():
                    if key in ("all_correct", "all_state_correct"):
                        continue
                    predicate = key.split(" ")[0]
                    model_answer = res[0]
                    state_correct = res[6]
                    if predicate not in predicate_stats:
                        predicate_stats[predicate] = {
                            "accuracy": 0,
                            "yes_accuracy": 0,
                            "yes_correct": 0,
                            "yes_total": 0,
                            "no_accuracy": 0,
                            "no_correct": 0,
                            "no_total": 0,
                        }
                    if model_answer == "yes":
                        if state_correct:
                            predicate_stats[predicate]["yes_correct"] += 1
                        predicate_stats[predicate]["yes_total"] += 1
                    elif model_answer == "no":
                        if not state_correct:
                            predicate_stats[predicate]["no_correct"] += 1
                        predicate_stats[predicate]["no_total"] += 1
                    else:
                        logger.warning(f"Unexpected answer {model_answer} for predicate {predicate}")
                        continue

                for action in results[problem]["action_results"]:
                    update_predicate_stats(predicate_stats, action.get("preconditions_results", {}))
                    update_predicate_stats(predicate_stats, action.get("effects_results", {}))

    action_accuracy = sum([problem_stats[problem]["action_correct"] for problem in problem_stats]) / sum(
        [problem_stats[problem]["action_total"] for problem in problem_stats]
    )
    fail_ratio = sum([problem_stats[problem]["failed"] for problem in problem_stats]) / len(problem_stats)

    for predicate in predicate_stats:
        predicate_stats[predicate]["correct"] = predicate_stats[predicate]["yes_correct"] + predicate_stats[predicate][
            "no_correct"
        ]
        predicate_stats[predicate]["total"] = predicate_stats[predicate]["yes_total"] + predicate_stats[predicate][
            "no_total"
        ]
        predicate_stats[predicate]["accuracy"] = (
            predicate_stats[predicate]["correct"] / predicate_stats[predicate]["total"]
            if predicate_stats[predicate]["total"] > 0
            else 0
        )

        predicate_stats[predicate]["yes_accuracy"] = (
            predicate_stats[predicate]["yes_correct"] / predicate_stats[predicate]["yes_total"]
            if predicate_stats[predicate]["yes_total"] > 0
            else 0
        )
        predicate_stats[predicate]["no_accuracy"] = (
            predicate_stats[predicate]["no_correct"] / predicate_stats[predicate]["no_total"]
            if predicate_stats[predicate]["no_total"] > 0
            else 0
        )

    logger.debug(f"Predicate stats: {predicate_stats}")
    predicate_accuracy = sum([predicate_stats[predicate]["correct"] for predicate in predicate_stats]) / sum(
        [predicate_stats[predicate]["total"] for predicate in predicate_stats]
    )
    macro_predicate_accuracy = (
        sum([predicate_stats[predicate]["accuracy"] for predicate in predicate_stats]) / len(predicate_stats)
        if len(predicate_stats) > 0
        else 0
    )

    return predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio
