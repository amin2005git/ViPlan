import copy
import os

from typing import Any, Dict, List, Optional, Tuple

from mloggers import Logger
from PIL.Image import Image

from viplan.code_helpers import parse_output

from enum import Enum

class FailureCase(str, Enum):
    ACTION_ILLEGAL = "action illegal"
    PRECOND_NOT_SATISFIED = "precond not satisfied"
    NON_VISIBLE_PRECOND_NOT_SATISFIED = "non-visible precond not satisfied"

ParsedVlmResult = Tuple[str, float, float, Optional[str], bool, Any, bool]
VlmResults = Dict[str, ParsedVlmResult]
EnumQuestion = Tuple[str, bool]
EnumResults = Dict[str, Tuple[str, str]]


def cast_to_yes_no(parsed_answer: str, logger: Logger) -> str:
    """
    Casts a parsed answer to a yes/no format.
    Args:
        parsed_answer (str): The answer to cast.
        logger (Logger): Logger to log debug information.
    Returns:
        str: The casted answer, either "yes" or "no". If the answer is empty or invalid, returns "invalid answer".
    """
    if parsed_answer is None:  # empty answer or invalid format
        return "invalid answer"

    if not parsed_answer.startswith("yes") and not parsed_answer.startswith("no"):
        for word in parsed_answer.split(" "):
            if word.startswith("yes"):
                parsed_answer = "yes"
                logger.debug(f"Found 'yes' in answer: {parsed_answer}")
                break
            elif word.startswith("no"):
                parsed_answer = "no"
                logger.debug(f"Found 'no' in answer: {parsed_answer}")
                break
    elif parsed_answer.startswith("yes"):
        parsed_answer = "yes"
    elif parsed_answer.startswith("no"):
        parsed_answer = "no"

    return parsed_answer


def _format_prompt_history(
    *,
    action_str: str,
    qa_pairs: List[Tuple[str, str]],
    case: FailureCase,
) -> str:
    lines: List[str] = []
    lines.append(
        "Previously, in the same state (image) as the one shown, you were asked these questions and provided the following answers:"
    )
    if len(qa_pairs) == 0:
        lines.append("(No questions were asked)")
    else:
        for question_text, model_answer in qa_pairs:
            lines.append(f"- Q: {question_text}")
            lines.append(f"  A: {model_answer}")
    lines.append("")
    if case is FailureCase.ACTION_ILLEGAL:
        lines.append(
            f"Based on these answers, the action '{action_str}' was attempted, but something went wrong. This is very likely to have been caused by one or more incorrect answers above. Keep this in mind while answering the following question."
        )
    elif case is FailureCase.PRECOND_NOT_SATISFIED:
         lines.append(
            f"Based on these answers, the previously-planned action '{action_str}' was called off, as at least one of its preconditions was judged invalid. This might have been the correct choice or a mistake. Keep this in mind while answering the following question."
        )
    elif case is FailureCase.NON_VISIBLE_PRECOND_NOT_SATISFIED:
         lines.append(
            f"Based on these answers, the action '{action_str}' was attempted, but something went wrong. This may have been due to an unobserved precondition not being met. Keep this in mind while answering the following question."
        )
    else:
        raise ValueError(f"Unknown failure case: {case}")
    return "\n".join(lines)
# 'action illegal', 'precond not satisfied', 'non-visible precond not satisfied'

def _extract_qa_pairs(
    questions: Dict[str, Tuple[str, Any]],
    results: Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]],
) -> List[Tuple[str, str]]:
    qa_pairs: List[Tuple[str, str]] = []
    for key, (question_text, _) in questions.items():
        if key not in results:
            continue
        qa_pairs.append((question_text, str(results[key][0])))
    return qa_pairs


def ask_vlm(
    questions: Dict[str, Tuple[str, str]],
    image: Image,
    model: Any,
    base_prompt: os.PathLike,
    logger: Logger,
    env: Any,
    prompt_history: Optional[str] = None,
    **kwargs,
) -> VlmResults:
    """
    Asks the VLM a set of questions and returns the answers.
    Args:
        questions (Dict[str, Tuple[str, str]]): A dictionary where keys are predicate names with arguments and values are tuples containing the question string and the expected answer (True/False).
        image (Image): The image to use for the VLM.
        model (Any): The VLM model to use for generating answers. Can be a VLLM or Transformers implementation from viplan.models.
        base_prompt (os.PathLike): The path to the base prompt file.
        logger (Logger): Logger to log debug information.
        env (Any): The environment simulator to use for grounding.
        **kwargs: Additional keyword arguments to pass to the model's generate method.
    Returns:
        Dict[str, Tuple[str, float, float, Optional[str], bool, Any, bool]]:
            - parsed_answer (str): The answer to the question, either "yes" or "no".
            - yes_prob (float): The probability of the answer being "yes" (does not work with CoT).
            - no_prob (float): The probability of the answer being "no" (does not work with CoT).
            - parsed_explanation (Optional[str]): The explanation of the answer, if available (e.g. for CoT).
            - answer_match (bool): Whether the answer matches the expected answer from the PDDL model.
            - original_output (Any): The original output from the model, for debugging purposes.
            - state_match (bool): Whether the answer matches the actual state of the environment.
    """

    base_prompt_text = open(base_prompt, "r").read()
    if prompt_history is not None and str(prompt_history).strip() != "":
        print("Using prompt history:\n", prompt_history)
        prompt_prefix = base_prompt_text.rstrip() + "\n\n" + str(prompt_history).strip() + "\n\n"
    else:
        print("No prompt history provided")
        prompt_prefix = base_prompt_text
    prompts = [prompt_prefix + q[0] for q in questions.values()]
    images = [image for _ in questions]

    outputs = model.generate(prompts=prompts, images=images, return_probs=True, **kwargs)

    results = {}
    for j, (key) in enumerate(questions.keys()):
        answer, yes_prob, no_prob = outputs[j]
        original_output = copy.deepcopy(answer)
        answer, tags_found = parse_output(answer, answer_tags=["answer", "explanation"])
        parsed_answer = answer["answer"] if tags_found and "answer" in answer else answer
        parsed_answer = (
            parsed_answer.strip().lower().rstrip(".,!?") if isinstance(parsed_answer, str) else parsed_answer
        )
        parsed_answer = cast_to_yes_no(parsed_answer, logger)

        parsed_explanation = answer["explanation"] if tags_found and "explanation" in answer else None

        logger.info(f"Q: {questions[key][0]}, A: {parsed_answer}, Yes: {yes_prob:.2f}, No: {no_prob:.2f}")
        if parsed_explanation is not None:
            logger.info(f"Explanation (CoT): {parsed_explanation}")

        # Answer match = the answer is what the PDDL model would expect -> if false, then replan
        answer_match = parsed_answer == "yes" if questions[key][1] else parsed_answer == "no"

        # State match = the answer is what is actually true in the environment
        # Can differ from PDDL if the action failed in the environment and the model correctly notices
        # e.g. block is dropped in the wrong column, PDDL expects it to be in the correct column but the model sees it dropped
        # -> state match is used to compute action accuracy, as it is the metric checking the model's actual performance

        predicate, args = key.split(" ")
        state_value = env.state[predicate][args]
        if (state_value and parsed_answer == "yes") or (not state_value and parsed_answer == "no"):
            state_match = True
        else:
            state_match = False
        logger.info(
            f"Actual predicate value: {predicate} {args} = {state_value}, model answer: {parsed_answer}, state match: {state_match}"
        )
        logger.info(
            f"PDDL expected value: {predicate} {args} = {questions[key][1]}, model answer: {parsed_answer}, answer match: {answer_match}"
        )

        results[key] = (parsed_answer, yes_prob, no_prob, parsed_explanation, answer_match, original_output, state_match)

    # All correct = all predicates are correct according to the PDDL model -> no replan needed
    # State match can only be used for metrics -> in the real world, the ground truth is not known
    results["all_correct"] = all([results[result][4] for result in results])
    results["all_state_correct"] = all(
        [results[k][6] for k in results if k not in ["all_correct", "all_state_correct"]]
    )

    return results


def get_enumeration_results(
    env: Any,
    model: Any,
    questions: Dict[str, EnumQuestion],
    base_prompt: os.PathLike,
    logger: Logger,
    batch_size: int = 64,
    prompt_history: Optional[str] = None,
) -> EnumResults:
    """
    Runs enumeration over a set of predicate questions, querying the VLM in batches and parsing the results.

    Args:
        env (Any): The environment simulator to use for rendering images.
        model (Any): The VLM model used for generating answers.
        questions (Dict[str, Tuple[str, bool]]): Dictionary mapping predicate keys to (question string, expected answer as bool).
        base_prompt (os.PathLike): Path to the base prompt file.
        logger (Logger): Logger for debug information.
        batch_size (int, optional): Number of questions to process per batch. Defaults to 64.
        prompt_history (Optional[str], optional): Previous action context to include in the prompt. Defaults to None.

    Returns:
        Dict[str, Tuple[str, str]]:
            - key: Predicate key string.
            - value: Tuple of (parsed model answer as 'yes'/'no', expected answer as 'yes'/'no').
    """
    responses = []
    base_prompt_text = open(base_prompt, "r").read()
    if prompt_history is not None and str(prompt_history).strip() != "":
        prompt_prefix = base_prompt_text.rstrip() + "\n\n" + str(prompt_history).strip() + "\n\n"
    else:
        prompt_prefix = base_prompt_text + "\n"
    q_list = [prompt_prefix + question[0] for question in questions.values()]

    for i in range(0, len(questions), batch_size):
        logger.info(f"Processing questions {i} to {min(i + batch_size, len(q_list))} of {len(q_list)}")
        batch_prompts = q_list[i : i + min(batch_size, len(q_list) - i)]
        img = env.render()
        batch_images = [img] * len(batch_prompts)
        batch_responses = model.generate(prompts=batch_prompts, images=batch_images, return_probs=True)
        responses.extend(batch_responses)

    assert len(responses) == len(q_list), "Some answers were not generated."

    results: Dict[str, Tuple[str, str]] = {}
    for i, (question, response) in enumerate(zip(questions, responses)):
        logger.debug(f"Question: {question}, Response: {response}")
        question_str = questions[question][0]
        answer = "yes" if questions[question][1] else "no"
        parsed, tags_found = parse_output(response[0], answer_tags=["answer", "explanation"])
        parsed_answer = parsed["answer"] if tags_found and "answer" in parsed else parsed
        parsed_answer = (
            parsed_answer.strip().lower().rstrip(".,!?") if isinstance(parsed_answer, str) else parsed_answer
        )
        parsed_answer = cast_to_yes_no(parsed_answer, logger)
        logger.debug(f"Parsed answer: {parsed_answer}")

        results[question] = (parsed_answer, answer.lower())

    return results
