import copy

from typing import Dict, Tuple

from unified_planning.environment import get_environment
from unified_planning.model import Problem


def update_problem(state: Dict[str, Dict[str, bool]], problem: Problem) -> Problem:
    """
    Creates a new problem initialized with the state the VLM thinks is currently true. Used for replanning online.
        Args:
            state (Dict[str, Dict[str, bool]]): The believed current state of the environment.
            problem (Problem): The original problem to update.
        Returns:
            Problem: A new problem with the initial values updated to match the current state.
    """

    def get_new_problem_fluent(new_problem, fluent):
        for new_fluent in new_problem.initial_values:
            if str(new_fluent) == str(fluent):
                return new_fluent
        return None

    new_problem = copy.deepcopy(problem)

    up_env = get_environment()
    expr_manager = up_env.expression_manager

    for fluent in problem.initial_values:

        name = fluent.fluent().name
        args = fluent.args
        args_str = ",".join([str(arg) for arg in args])
        value = problem.initial_values[fluent].is_true()
        # TODO: solve KeyError: 'ontop' or 'inside' -> issue is in state not having the fluent name
        # Track state and enforce that it always has a key for every fluent na,e in the problem.initial_values

        # Quickfix / failsafe for now
        if name not in state.keys():
            continue

        if args_str not in state[name]:
            continue
        state_value = state[name][args_str]
        if value != state_value:
            assert problem.initial_values[fluent].is_bool_constant()
            new_fluent = get_new_problem_fluent(new_problem, fluent)
            if new_fluent is None:
                raise ValueError(f"Fluent {fluent} not found in new_problem")

            new_problem.initial_values[new_fluent] = (
                expr_manager.true_expression if state_value else expr_manager.false_expression
            )
            # print(f"Updating {new_fluent} from {problem.initial_values[fluent]} to {new_problem.initial_values[new_fluent]}")

    return new_problem


def update_vlm_state(
    vlm_state: Dict[str, Dict[str, bool]],
    results: Dict[str, Tuple[str, float, float, str, bool, object, bool]],
) -> Tuple[Dict[str, Dict[str, bool]], list]:
    """
    Updates the VLM state dictionary based on the results from the VLM model.
    Args:
        vlm_state (Dict[str, Dict[str, bool]]): The current state of the environment as believed by the VLM. Needs to be a copy.deepcopy() of the original state.
        results (Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]]):
            The results returned from the VLM model for each predicate question.
    Returns:
        Tuple[Dict[str, Dict[str, bool]], List[str]]:
            - Updated VLM state dictionary.
            - List of changed predicates as strings for logging.
    """
    changed = []
    for key, result in results.items():
        if key in ("all_correct", "all_state_correct", "updated_non_visible_preds"):
            continue
        pred, args = key.split(" ")
        assert result[0] == "yes" or result[0] == "no", f"VLM gave unexpected answer {result[0]}"
        new_value = True if result[0] == "yes" else False
        if vlm_state[pred][args] != new_value:
            vlm_state[pred][args] = new_value
            changed_str = f"{pred} {args} to {new_value}"
            changed.append(changed_str)

    return vlm_state, changed
