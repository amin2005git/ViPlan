import copy
import os

from typing import Any, Dict, List, Optional, Tuple

from mloggers import Logger
from unified_planning.model import FNode
from unified_planning.plans import ActionInstance

from viplan.planning.planning_simulator import PlanningSimulator

from viplan.experiments.vlm_grounder.questions import (
    get_effects_predicates,
    get_preconditions_predicates,
    get_question_preds,
    get_questions,
)
from viplan.experiments.vlm_grounder.state import update_vlm_state
from viplan.experiments.vlm_grounder.vlm_query import (
    VlmResults,
    _extract_qa_pairs,
    _format_prompt_history,
    ask_vlm,
    FailureCase
)


def check_preconditions(
    env: PlanningSimulator,
    preconditions: FNode,
    grounded_args: Dict[str, str],
    model: Any,
    base_prompt: os.PathLike,
    logger: Logger,
    predicate_questions: Dict[str, str],
    argument_aliases: Optional[Dict[str, str]] = None,
    prompt_history: Optional[str] = None,
) -> Tuple[VlmResults, Dict[str, Tuple[bool, bool, bool]], List[Tuple[str, str]]]:
    """
    Checks preconditions by querying the VLM for visible predicates and comparing non-visible predicates against the environment state.

    Args:
        env (PlanningSimulator): The planning environment.
        preconditions (FNode): The preconditions node to extract predicates from.
        grounded_args (Dict[str, str]): Mapping of argument names to grounded values.
        model (Any): The VLM model used for generating answers.
        base_prompt (os.PathLike): Path to the base prompt file.
        logger (Logger): Logger for debug information.
        predicate_questions (Dict[str, str]): Mapping of predicate names to their corresponding question templates.
        argument_aliases (Optional[Dict[str, str]]): Mapping of argument names to their aliases for question readability.

    Returns:
        Tuple[
            Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]],
            Dict[str, Tuple[bool, bool, bool]]
        ]:
            - results: VLM results for visible predicates.
            - non_visible_results: Comparison results for non-visible predicates.
    """
    precondition_preds = get_preconditions_predicates(env, preconditions, grounded_args)
    logger.debug(f"Precondition predicates: {precondition_preds}")
    visible_preds = env.visible_predicates
    logger.debug(f"Visible predicates: {visible_preds}")
    question_preds, non_visible_preds = get_question_preds(precondition_preds, visible_preds)
    logger.debug(f"Non visible predicates: {non_visible_preds}")
    logger.debug(f"Question predicates: {question_preds}")

    questions = get_questions(question_preds, predicate_questions, argument_aliases)

    if len(questions) == 0:
        logger.warning("No questions to ask VLM")
        results = {}
    else:
        results = ask_vlm(
            questions, env.render(), model, base_prompt, logger, env, prompt_history=prompt_history
        )
        logger.debug(f"Precondition VLM results: {results}")

    qa_pairs = _extract_qa_pairs(questions, results) if len(questions) > 0 else []

    # Check non visible predicates against vlm_state
    non_visible_results = {}
    for predicate in non_visible_preds:
        key = list(predicate.keys())[0]
        pddl_expected_value = predicate[key]
        predicate = key.split(" ")[0]
        args = ",".join(key.split(" ")[1:])
        vlm_state_value = env.state.get(predicate, {}).get(args, False)
        non_visible_results[key] = (vlm_state_value == pddl_expected_value, vlm_state_value, pddl_expected_value)
        if vlm_state_value != pddl_expected_value:
            logger.warning(
                f"Non visible predicate {predicate} {args} does not match PDDL model: {vlm_state_value} != {pddl_expected_value}"
            )

    non_visible_results["all_correct"] = all([non_visible_results[k][0] for k in non_visible_results])

    return results, non_visible_results, qa_pairs


def check_effects(
    env: PlanningSimulator,
    vlm_state: Dict[str, Dict[str, bool]],
    effects: List[FNode],
    grounded_args: Dict[str, str],
    model: Any,
    base_prompt: os.PathLike,
    previous_state: Dict[str, Dict[str, bool]],
    logger: Logger,
    predicate_questions: Dict[str, str],
    argument_aliases: Optional[Dict[str, str]] = None,
    prompt_history: Optional[str] = None,
) -> Tuple[VlmResults, Dict[str, Dict[str, bool]], List[Tuple[str, str]]]:
    """
    Checks effects by querying the VLM for visible predicates and updating non-visible predicates in the VLM state.

    Args:
        env (PlanningSimulator): The planning environment.
        vlm_state (Dict[str, Dict[str, bool]]): The current state as believed by the VLM.
        effects (List[FNode]): List of effect nodes to extract predicates from.
        grounded_args (Dict[str, str]): Mapping of argument names to grounded values.
        model (Any): The VLM model used for generating answers.
        base_prompt (os.PathLike): Path to the base prompt file.
        previous_state (Dict[str, Dict[str, bool]]): State before effects are applied.
        logger (Logger): Logger for debug information.
        predicate_questions (Dict[str, str]): Mapping of predicate names to their corresponding question templates.
        argument_aliases (Optional[Dict[str, str]]): Mapping of argument names to their aliases for question readability.

    Returns:
        Tuple[
            Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]],
            Dict[str, Dict[str, bool]]
        ]:
            - results: VLM results for visible predicates.
            - vlm_state: Updated VLM state dictionary.
    """
    effect_preds = get_effects_predicates(env, effects, grounded_args, previous_state)
    logger.debug(f"Effect predicates: {effect_preds}")
    visible_preds = env.visible_predicates
    logger.debug(f"Visible predicates: {visible_preds}")
    question_preds, non_visible_preds = get_question_preds(effect_preds, visible_preds)
    logger.debug(f"Non visible predicates: {non_visible_preds}")
    logger.debug(f"Question predicates: {question_preds}")

    questions = get_questions(question_preds, predicate_questions, argument_aliases)
    if len(questions) == 0:
        logger.warning("No questions to ask VLM")
        results = {}
    else:
        results = ask_vlm(
            questions, env.render(), model, base_prompt, logger, env, prompt_history=prompt_history
        )

    qa_pairs = _extract_qa_pairs(questions, results) if len(questions) > 0 else []

    # Update vlm_state with non visible preds using the PDDL expected value
    updated_non_visible_preds = {}
    for predicate in non_visible_preds:
        key = list(predicate.keys())[0]
        pddl_expected_value = predicate[key]
        predicate = key.split(" ")[0]
        args = ",".join(key.split(" ")[1:])
        updated_non_visible_preds[f"{predicate} {args}"] = {
            "before": vlm_state[predicate][args] if args in vlm_state[predicate] else None,
            "after": pddl_expected_value,
        }
        logger.debug(f"Updating vlm_state for {predicate} {args} to {pddl_expected_value}")
        vlm_state[predicate][args] = pddl_expected_value

    results["updated_non_visible_preds"] = updated_non_visible_preds
    return results, vlm_state, qa_pairs


def check_action(
    env: PlanningSimulator,
    action: ActionInstance,
    vlm_state: Dict[str, Dict[str, bool]],
    model: Any,
    base_prompt: os.PathLike,
    logger: Logger,
    predicate_questions: Dict[str, str],
    argument_aliases: Optional[Dict[str, str]] = None,
    prompt_history: Optional[str] = None,
    include_prompt_history: bool = False,
) -> Tuple[
    bool,
    VlmResults,
    Dict[str, Tuple[bool, bool, bool]],
    Optional[VlmResults],
    bool,
    Any,
    Optional[str],
]:
    """
    Checks the execution of a single action by verifying preconditions and effects using the VLM and environment state.

    Args:
        env (PlanningSimulator): The planning environment.
        action (Any): The action to execute.
        vlm_state (Dict[str, Dict[str, bool]]): The current state as believed by the VLM.
        model (Any): The VLM model used for generating answers.
        base_prompt (os.PathLike): Path to the base prompt file.
        logger (Logger): Logger for debug information.
        predicate_questions (Dict[str, str]): Mapping of predicate names to their corresponding question templates.
        argument_aliases (Optional[Dict[str, str]]): Mapping of argument names to their aliases for question readability.

    Returns:
        Tuple[
            bool,  # all_correct: Whether both preconditions and effects are correct according to the PDDL model.
            Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]],  # preconditions_results: VLM results for visible preconditions.
            Dict[str, Tuple[bool, bool, bool]],  # non_visible_precond_results: Comparison results for non-visible preconditions.
            Optional[Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]]],  # effects_results: VLM results for visible effects.
            bool,  # all_state_correct: Whether both preconditions and effects match the actual environment state.
            Any,  # info: Additional info from the environment after applying the action.
        ]
    """
    preconditions = action.action.preconditions
    effects = action.action.effects
    grounded_params = {param.name: str(value) for param, value in zip(action.action.parameters, action.actual_parameters)}
    previous_state = copy.deepcopy(env.state)
    logger.info("Environment state before action\n" + str(env))

    history_for_next: Optional[str] = None

    preconditions_results, non_visible_precond_results, precond_qa_pairs = check_preconditions(
        env,
        preconditions,
        grounded_params,
        model,
        base_prompt,
        logger,
        predicate_questions,
        argument_aliases,
        prompt_history=prompt_history,
    )
    if not non_visible_precond_results["all_correct"]:
        logger.warning("Non visible preconditions not satisfied")
        if include_prompt_history:
            history_for_next = _format_prompt_history(
                action_str=str(action), qa_pairs=precond_qa_pairs, case=FailureCase.NON_VISIBLE_PRECOND_NOT_SATISFIED
            )
        return (
            False,
            preconditions_results,
            non_visible_precond_results,
            None,
            False,
            None,
            history_for_next,
        )

    vlm_state, changed = update_vlm_state(vlm_state, preconditions_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after preconditions:", changed)

    # If the preconditions are not satisfied according to the PDDL model, the action can not be taken
    if "all_correct" in preconditions_results and not preconditions_results["all_correct"]:
        logger.warning("Preconditions not satisfied")
        if include_prompt_history:
            history_for_next = _format_prompt_history(
                action_str=str(action), qa_pairs=precond_qa_pairs, case=FailureCase.PRECOND_NOT_SATISFIED
            )
        return (
            False,
            preconditions_results,
            non_visible_precond_results,
            None,
            False,
            None,
            history_for_next,
        )

    legal, info = env.apply_action(plan_action=action)
    # VLM thought the action was legal, but it was not
    if not legal:
        logger.warning("Action was not legal")
        if include_prompt_history:
            history_for_next = _format_prompt_history(
                action_str=str(action), qa_pairs=precond_qa_pairs, case=FailureCase.ACTION_ILLEGAL
            )
        return (
            False,
            preconditions_results,
            non_visible_precond_results,
            None,
            False,
            info,
            history_for_next,
        )

    logger.info("Environment state after action\n" + str(env))

    effects_results, vlm_state, effects_qa_pairs = check_effects(
        env,
        vlm_state,
        effects,
        grounded_params,
        model,
        base_prompt,
        previous_state,
        logger,
        predicate_questions,
        argument_aliases,
        prompt_history=prompt_history,
    )
    vlm_state, changed = update_vlm_state(vlm_state, effects_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after effects:", changed)

    precond_all_correct = preconditions_results["all_correct"] if "all_correct" in preconditions_results else True
    effects_all_correct = effects_results["all_correct"] if "all_correct" in effects_results else True
    all_correct = precond_all_correct and effects_all_correct

    # Effects state correct can be different from all_correct if there was a failure in the environment and the VLM detected it
    precond_state_correct = (
        preconditions_results["all_state_correct"] if "all_state_correct" in preconditions_results else True
    )
    effects_state_correct = effects_results["all_state_correct"] if "all_state_correct" in effects_results else True
    all_state_correct = precond_state_correct and effects_state_correct

    # Note: history_for_next is only set on precondition failures, not effect failures
    return (
        all_correct,
        preconditions_results,
        non_visible_precond_results,
        effects_results,
        all_state_correct,
        info,
        history_for_next,
    )
