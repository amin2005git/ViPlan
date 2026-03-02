from typing import List, Dict, Tuple, Optional

from unified_planning.model import FNode

from viplan.planning.planning_simulator import PlanningSimulator


def get_questions(
    predicates: List[dict] | Dict[str, Dict[str, bool]],
    predicate_questions: Dict[str, str],
    argument_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Given a list of predicates, return a dictionary mapping each predicate to a question and its value
    Args:
        predicates (List[dict] | Dict[str, Dict[str, bool]]): A dictionary or list of predicates where keys are predicate names and values are their arguments. These are returned from either "get_preconditions_predicates" or "get_effects_predicates".
        predicate_questions (Dict[str, str]): A dictionary mapping predicate names to their corresponding question templates.
        argument_aliases (Optional[Dict[str, str]]): A dictionary mapping argument names to their aliases for question readability. They are replaced by exact match "e.g. 'r' to 'red block'".
    Returns:
        Dict[str, Tuple[str, str]]: A dictionary where keys are predicate names with arguments and values are tuples containing the question string and the correct answer (True/False).
    """

    def replace_args(args: List[str]) -> List[str]:
        return [argument_aliases.get(arg, arg) for arg in args] if argument_aliases else args

    questions = {}
    if isinstance(predicates, dict):
        for predicate, arg_dict in predicates.items():
            if predicate not in predicate_questions:
                raise ValueError(f"Unknown predicate '{predicate}'")

            template = predicate_questions[predicate]

            for args_str, expected in arg_dict.items():
                args = args_str.split(",")
                readable_args = replace_args(args)
                question_key = f"{predicate} {args_str}"
                question_text = template.format(*readable_args)
                questions[question_key] = ("Question: " + question_text, expected)
    else:
        for pred in predicates:
            key = list(pred.keys())[0]
            predicate, args_str = key.split(" ")
            args = args_str.split(",")
            expected = pred[key]

            if predicate not in predicate_questions:
                raise ValueError(f"Unknown predicate '{predicate}'")

            template = predicate_questions[predicate]
            readable_args = replace_args(args)
            question_key = f"{predicate} {args_str}"
            question_text = template.format(*readable_args)
            questions[question_key] = ("Question: " + question_text, expected)

    return questions


def get_predicates_for_question(
    env: PlanningSimulator,
    node: FNode,
    grounded_args: Dict[str, str],
    top_level=True,
    default_value=True,
) -> List[Dict[str, bool]]:
    """
    Recursively extracts predicates from a node for question generation.
    Args:
        env (PlanningSimulator): The environment simulator to use for grounding.
        node (FNode): The node to extract predicates from.
        grounded_args (Dict[str, str]): A dictionary mapping argument names to their grounded values.
        top_level (bool): Whether this is the top-level call. Used to skip "or" nodes.
        default_value (bool): The default value to assume for preconditions when not specified.
    Returns:
        List[Dict[str, bool]]: A list of dictionaries where each dictionary contains a predicate name with its arguments as keys and their boolean values as values.
    """
    result = []

    if not node.is_fluent_exp():

        if node.is_or():
            # We skip all "or" nodes (which only happens once in our domain) as our method asks questions independently to the VLM, and thus a disjunction is not possible
            # Since the only disjunction (in navigate-to) is by definition never visible by the VLM (as it asks for objects inside containers), we can ignore it
            return {}

        if node.is_not():
            child = node.args[0]
            exps = get_predicates_for_question(env, child, grounded_args, False, default_value)
            for exp in exps:
                result.append({k: not v for k, v in exp.items()})
        elif node.is_forall():
            assert len(node.variables()) == 1, "Only single forall supported"
            var = node.variables()[0]

            for value in env.all_objects[str(var.type)]:
                if str(value) in grounded_args.values():
                    continue
                grounded_args[var.name] = str(value)
                exps = get_predicates_for_question(env, node.args[0], grounded_args, False, default_value)
                result.extend(exps)

        else:
            for child in node.args:
                exps = get_predicates_for_question(env, child, grounded_args, False, default_value)
                result.extend(exps)

    elif node.is_fluent_exp():
        fluent_name = node.fluent().name
        arg_names = [str(arg) for arg in node.args]
        actual_args = [str(grounded_args[arg]) for arg in arg_names]
        args_key = ",".join(actual_args)
        key = fluent_name + " " + args_key
        value = default_value  # by default assume precondition is asking for predicate to be default_value (for effects it can also be False)
        bool_value = value if isinstance(value, bool) else value.is_true()
        result = {key: bool_value}

    else:
        raise ValueError("Unknown node type", node)

    return [result] if type(result) is dict else result


def get_effect_predicates(
    env: PlanningSimulator,
    effect: FNode,
    grounded_args: Dict[str, str],
    previous_state: Dict[str, Dict[str, bool]],
) -> List[Dict[str, bool]]:
    """
    Extracts predicates from a single effect node for question generation.
    Args:
        env (PlanningSimulator): The environment simulator to use for grounding.
        effect (FNode): The effect node to extract predicates from.
        grounded_args (Dict[str, str]): A dictionary mapping argument names to their grounded values.
        previous_state (Dict[str, Dict[str, bool]]): The state before the effect is applied, used for conditional effects.
    Returns:
        List[Dict[str, bool]]: A list of dictionaries where each dictionary contains a predicate name with its arguments as keys and their boolean values as values.
    """
    all_preds = []

    if effect.is_forall():
        vars_ = effect.forall
        # Support single or double forall
        if len(vars_) == 1:
            var = vars_[0]
            for value in env.all_objects[str(var.type)]:
                grounded_args[var.name] = str(value)
                # If conditional, check against previous state
                if effect.is_conditional():
                    if not env._check_value(effect.condition, grounded_args, previous_state):
                        continue
                preds = get_predicates_for_question(
                    env, effect.fluent, grounded_args, top_level=False, default_value=effect.value
                )
                if isinstance(preds, dict):
                    all_preds.append(preds)
                else:
                    all_preds.extend(preds)
        elif len(vars_) == 2:
            var1, var2 = vars_
            for v1 in env.all_objects[str(var1.type)]:
                grounded_args[var1.name] = str(v1)
                for v2 in env.all_objects[str(var2.type)]:
                    grounded_args[var2.name] = str(v2)
                    if effect.is_conditional():
                        if not env._check_value(effect.condition, grounded_args, previous_state):
                            continue
                    preds = get_predicates_for_question(
                        env, effect.fluent, grounded_args, top_level=False, default_value=effect.value
                    )
                    if isinstance(preds, dict):
                        all_preds.append(preds)
                    else:
                        all_preds.extend(preds)
        else:
            raise NotImplementedError("Only up to 2 nested foralls are supported")

    elif effect.is_conditional():
        condition = effect.condition
        # For conditional effects, check condition against previous_state.
        if env._check_value(condition, grounded_args, previous_state):
            preds = get_predicates_for_question(
                env, effect.fluent, grounded_args, top_level=False, default_value=effect.value
            )
            if isinstance(preds, dict):
                all_preds.append(preds)
            else:
                all_preds.extend(preds)

    else:
        preds = get_predicates_for_question(
            env, effect.fluent, grounded_args, top_level=False, default_value=effect.value
        )
        if isinstance(preds, dict):
            all_preds.append(preds)
        else:
            all_preds.extend(preds)

    return all_preds


def get_preconditions_predicates(
    env: PlanningSimulator,
    preconditions: FNode,
    grounded_args: Dict[str, str],
) -> List[Dict[str, bool]]:
    """
    Extracts predicates from preconditions for question generation.
    Args:
        env (PlanningSimulator): The environment simulator to use for grounding.
        preconditions (FNode): The preconditions node to extract predicates from.
        grounded_args (Dict[str, str]): A dictionary mapping argument names to their grounded values.
    Returns:
        List[Dict[str, bool]]: A list of dictionaries where each dictionary contains a predicate name with its arguments as keys and their boolean values as values.
    """
    precond_list = []
    for precondition in preconditions:
        if precondition.is_and():
            precond_list.extend(precondition.args)

    preconditions_predicates = []
    for precondition in precond_list:
        preds = get_predicates_for_question(env, precondition, grounded_args)
        preconditions_predicates.extend(preds)
    return preconditions_predicates


def get_effects_predicates(
    env: PlanningSimulator,
    effects: List[FNode],
    grounded_args: Dict[str, str],
    previous_state: Dict[str, str],
) -> List[Dict[str, bool]]:
    """
    Extracts predicates from effects for question generation.
    Args:
        env (PlanningSimulator): The environment simulator to use for grounding.
        effects (List[FNode]): A list of effect nodes to extract predicates from.
        grounded_args (Dict[str, str]): A dictionary mapping argument names to their grounded values.
        previous_state (Dict[str, str]): The state before the effects are applied, used for conditional effects.
    Returns:
        List[Dict[str, bool]]: A list of dictionaries where each dictionary contains a predicate name with its arguments as keys and their boolean values as values.
    """
    all_preds = []
    for effect in effects:
        preds = get_effect_predicates(env, effect, grounded_args, previous_state)
        all_preds.extend(preds)
    return all_preds


def get_question_preds(
    predicates: List[Dict[str, bool]],
    visible_preds: Dict[str, Dict[str, bool]],
) -> Tuple[List[Dict[str, bool]], List[Dict[str, bool]]]:
    """
    Filters predicates into those that are visible in the environment (and can be asked to the VLM) and those that are not.
    Args:
        predicates (List[Dict[str, bool]]): A list of predicates where each predicate is a dictionary with keys as predicate names and values as their arguments.
        visible_preds (Dict[str, Dict[str, bool]]): A dictionary mapping predicate names to their arguments that are visible in the environment.
    Returns:
        Tuple[List[Dict[str, bool]], List[Dict[str, bool]]]:
            - List of predicates that are visible.
            - List of predicates that are not visible.
    """
    question_preds = []
    non_visible_preds = []
    for predicate_dict in predicates:
        key = list(predicate_dict.keys())[0]
        predicate = key.split(" ")[0]
        args = key.split(" ")[1:]
        if predicate in visible_preds and ",".join(args) in visible_preds[predicate]:
            question_preds.append(predicate_dict)
        else:
            non_visible_preds.append(predicate_dict)

    return question_preds, non_visible_preds
