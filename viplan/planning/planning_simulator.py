from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from PIL import Image
from unified_planning.model import Problem
from unified_planning.plans.plan import ActionInstance


class PlanningSimulator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        self.state = None
        self.img: Image.Image = None
        self.supports_full_enumeration = True  # Default to True, should be overridden if the environment is partially observable and all predicates are not visible in the initial state
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}

    @abstractmethod
    def reset(self, *args, **kwargs) -> bool:
        """Resets the simulator state."""
        pass

    @abstractmethod
    def apply_action(self, plan_action: ActionInstance, *args, **kwargs) -> Tuple[bool, str]:
        """Apply an action to the simulator. Actions are instances of ActionInstance, other arguments can be used for additional context depending on the simulator."""
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Image.Image:
        """Render the current simulator state as an image."""
        pass
    
    def _add_missing_keys(self, state):
        for fluent in self.problem.initial_values:
            name = fluent.fluent().name 
            if name not in state.keys():
                state[name] = {}
        return state

    @property
    def goal_fluents(self):
        goal_predicates = self.problem.goals[0] # Assuming there are just a set of "ands" in the goal
        assert goal_predicates.is_and() or goal_predicates.is_fluent_exp(), "Goal structure was not a simple conjunction of fluents"
        
        if goal_predicates.is_and():
            for goal in goal_predicates.args:
                if goal.is_not():
                    goal = goal.args[0]
                assert goal.is_fluent_exp(), "Goal structure was not a simple conjunction of fluents"
            goal_fluents = goal_predicates.args
        else:
            goal_fluents = [goal_predicates]

        print("goal_fluents: ", goal_fluents)
        return goal_fluents
            
    @property
    def goal_reached(self):
        goal_fluents = self.goal_fluents
        self.logger.debug(f"Goal fluents: {goal_fluents}")
        self.logger.debug(f"State: {self.state}")
        if self.state is None:
            self.logger.error("State is None, cannot check goal reached")
            return False
            
        # Enforce here that all fluents exist as keys, even if just as empty dictionaries
        # should avoid issues in line `sat = self.state[fluent_name][",".join(fluent_args)] == value`
        self.state = self._add_missing_keys(self.state)
        
        for fluent in goal_fluents:
            value = True
            if fluent.is_not():
                fluent = fluent.args[0]
                value = False
            fluent_name = fluent.fluent().name
            fluent_args = [a.object().name for a in fluent.args]
            
            # Safeguard for missing keys in the state
            state_value = self.state.get(fluent_name, {}).get(",".join(fluent_args), None)
            if state_value is None:
                self.logger.warning(f"Goal fluent {str(fluent)} not found in state")
                return False
            sat = state_value == value
            if not sat:
                self.logger.debug(f"Goal fluent {str(fluent)} not satisfied")
                return False
        
        self.logger.info("Goal reached")
        return True

    # Recursively check the truth value of a node, which can contain either a fluent (predicate with arguments) or a logical operator
    # The state argument is needed because when applying effects the truth value of a fluent needs to be checked from the state BEFORE the effect was applied, since self.state will have incomplete updates
    def _check_value(self, node, grounded_args, state=None):
        if state is None:
            state = self._add_missing_keys(self.state) # add_missing_keys added for extra safety
        if node.is_and():
            return all([self._check_value(arg, grounded_args, state) for arg in node.args])
        elif node.is_or():
            return any([self._check_value(arg, grounded_args, state) for arg in node.args])
        elif node.is_not():
            return not self._check_value(node.args[0], grounded_args, state)
        elif node.is_equals():
            arg1, arg2 = node.args
            if arg1.is_parameter_exp() or arg1.is_variable_exp() and arg2.is_parameter_exp() or arg2.is_variable_exp():
                return grounded_args[str(arg1)] == grounded_args[str(arg2)]
            else:
                return self._check_value(arg1, grounded_args, state) == self._check_value(arg2, grounded_args, state) 
        elif node.is_fluent_exp():
            fluent_name = node.fluent().name
            arg_names = [str(arg) for arg in node.args]
            args = [grounded_args[arg] for arg in arg_names]
            if not ','.join(args) in state[fluent_name]:
                # This is for example on(cabinet1, cabinet1) which should be false but is not in the state
                return False
            value = state[fluent_name][','.join(args)]
            return value if isinstance(value, bool) else value.is_true()
        else:
            raise ValueError("This node type is not implemented in PlanningSimulator, please report this issue or add an implementation.", node)

    # Optional (only implemented if needed, otherwise returns None)
    @property
    def priviledged_predicates(self) -> Dict[str, set] | None:
        return None

    @property
    def visible_predicates(self) -> Dict[str, Dict[str, Any]]:
        return self.state # Should be overridden for partially observable environments
