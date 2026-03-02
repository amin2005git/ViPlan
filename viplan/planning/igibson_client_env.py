import io
import re
import json
import base64
import requests
from unified_planning.model import Problem
from unified_planning.plans.plan import ActionInstance

from PIL import Image

from viplan.code_helpers import get_logger
from viplan.planning.planning_simulator import PlanningSimulator

class iGibsonClient(PlanningSimulator):
    
    def __init__(self, 
                 task: str,
                 scene_id: str,
                 base_url: str,
                 problem: Problem,
                 instance_id: int = 0,
                 logger = get_logger('info'),
    ):
        
        super().__init__(problem)
        self.task = task
        self.scene_id = scene_id
        self.instance_id = instance_id
        self.problem = problem
        self.base_url = base_url
        self.logger = logger
        self.supports_full_enumeration = False # iGibson is a partially observable environment, so we cannot enumerate all predicates in the initial state
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}
        
        self.reset()
        self.state, self.img = self._get_state_and_img()
        
    def __str__(self):
        return str(self.state)
    
    def reset(self):
        # Update all_objects in case problem has changed
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}

        payload = {
            "task": self.task,
            "scene_id": self.scene_id,
            "instance_id": self.instance_id
        }
        
        response = requests.post(f"{self.base_url}/reset", json=payload)  
        
        if response.status_code == 200:
            data = response.json()
            self.logger.info(f"\nReset successful: {data['success']}")
            
            # Update the state and image after reset
            self.state, self.img = self._get_state_and_img()
            if self.state is None or self.img is None:
                self.logger.error("Failed to get initial state or image after reset")
                return False
            
            return True
        else:
            self.logger.error(f"Reset failed with status code {response.status_code}")
            self.logger.error(response.text)
            return False
        
    def _get_state_and_img(self):
        response = requests.get(f"{self.base_url}/get_state")
        
        if response.status_code == 200:
            data = response.json()
            state = data['symbolic_state']

            # Enforce here that all fluents exist as keys, even if just as empty dictionaries
            state = self._add_missing_keys(state)
            
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data))
            
            return state, img
        else:
            self.logger.error(f"Get state failed with status code {response.status_code}")
            self.logger.error(response.text)
            return None, None
        
    @property
    def priviledged_predicates(self):
        
        if self.state is not None:
            inside_preds = {k: v for k, v in self.state.get('inside', {}).items() if v}
            reachable_preds = {k: v for k, v in self.state.get('reachable', {}).items() if v}
            
            inside_keys = [key.split(',')[0] for key in inside_preds.keys()]
            inside_containers = [key.split(',')[1] for key in inside_preds.keys()]
            reachable_keys = list(reachable_preds.keys())
            inside_but_not_reachable = {'inside': set([(inside_keys[i], inside_containers[i]) for i in range(len(inside_keys)) if inside_keys[i] not in reachable_keys])}
            return inside_but_not_reachable
        else:
            self.logger.error("State is None, cannot get priviledged predicates")
            return None
    
    def _get_visible_objects(self):
        response = requests.get(f"{self.base_url}/get_visible_objects")
        
        if response.status_code == 200:
            data = response.json()
            visible_objects = data['objects']
            return visible_objects
        else:
            self.logger.error(f"Get visible objects failed with status code {response.status_code}")
            self.logger.error(response.text)
            return None
        
    @property
    def visible_predicates(self):
        visible_objects = self._get_visible_objects()
        if visible_objects is not None:
            visible_preds = {}
            for predicate in self.state:
                for args in self.state[predicate].keys():
                    if all([arg in visible_objects for arg in args.split(',')]):
                        if predicate not in visible_preds:
                            visible_preds[predicate] = {}
                        visible_preds[predicate][args] = self.state[predicate][args]
            return visible_preds
        else:
            self.logger.error("Failed to get visible objects, cannot get visible predicates")
            return None
        
    def render(self):
        # Keep the render function for compatibility with the other envs (blocksworld)
        return self.img
    
    def apply_action(self, plan_action: ActionInstance):

        action_name = plan_action.action.name
        params = [str(p) for p in plan_action.actual_parameters]

        payload = {
            "action": action_name,
            "params": params
        }
        
        response = requests.post(f"{self.base_url}/execute_action", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.logger.info(f"\nAction legal: {data['success']}")
            self.logger.info(f"\nAction execution successful: {data['info']}")
            
            # Update the state and image after action execution
            # self.state, self.img = self._get_state_and_img() # shouldn't be needed
            state = data['symbolic_state']
            self.state = self._add_missing_keys(state) 
            img_data = base64.b64decode(data['image'])
            self.img = Image.open(io.BytesIO(img_data))
            
            if self.state is None or self.img is None:
                self.logger.error("Failed to get state or image after action execution")
                return False, data['info']
            return data['success'], data['info']
        if response.status_code == 400:
            error_detail = response.json().get("detail", "Invalid request")
            self.logger.warning(f"Action rejected by server: {error_detail}")
            return False, error_detail
        else:
            self.logger.error(f"Action execution failed with status code {response.status_code}")
            self.logger.error(response.text)
            return False, f'server returned {response.status_code}'