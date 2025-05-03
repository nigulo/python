import numpy as np
import numpy.random as random
from enum import IntEnum

from typing import List, Dict, Set, Tuple, Callable
import types

class DP:
    
    #def __init__(self, 
    #             states: Set[int], 
    #             actions: Dict[int, Set[int]] | Set[int] | Callable[int, Set[int]], 
    #             transitions: Dict[Tuple[int, int], List[Tuple[int, float, float]] | Callable[[int, int], List[Tuple[int, float, float]]]], 
    def __init__(self, 
                 states, 
                 actions, 
                 transitions,
                 after_states=None):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.after_states = after_states
        
    def solve(self, gamma=0.9, n_iter=10000, eps=1e-7, random_initial_policy=False, pi_or_v_iter=True):
        v: Dict[int, float] = {}
        q: Dict[Tuple[int, int], float] = {}        
        pi: Dict[int, int] = {}

        for s in self.states:
            actions = list(self._get_actions(s))
            pi[s] = random.choice(actions) if random_initial_policy else actions[0]

        best_policy = False
        i = 0
        while not best_policy and i < n_iter:
            i += 1
            delta = eps
            while delta >= eps:
                delta = 0
                for s in self.states if not self.after_states else self.after_states:
                    if pi_or_v_iter:
                        a = pi[s]
                        v_s_max = 0
                        for s_prime, r, p in self._get_transitions(s, a, []):
                            if p > 0:
                                v_s_max += p*(r + gamma*v.get(s_prime, 0))
                    else:
                        a_max = pi[s]
                        v_s_max = -np.inf
                        for a in self._get_actions(s):
                            v_s = 0
                            for s_prime, r, p in self._get_transitions(s, a, []):
                                if p > 0:
                                    v_s += p*(r + gamma*v.get(s_prime, 0))
                            if v_s > v_s_max:
                                v_s_max = v_s
                                a_max = a
                        pi[s] = a_max
                    delta = max(delta, abs(v_s_max - v.get(s, 0)))
                    v[s] = v_s_max
            
            best_policy = True
            if not pi_or_v_iter:
                break

            for s in self.states:
                q_s_a_max = -np.inf
                a_max = pi[s]
                for a in self._get_actions(s):
                    q_s_a = 0
                    for s_prime, r, p in self._get_transitions(s, a, []):
                        if p > 0:
                            q_s_a += p*(r + gamma*v.get(s_prime, 0))
                    q[(s, a)] = q_s_a
                    if q_s_a > q_s_a_max:
                        q_s_a_max = q_s_a
                        a_max = a
                        if self.after_states and s not in self.after_states:
                            v[s] = q_s_a_max
                if q[(s, a_max)] != q[(s, pi[s])]:
                    pi[s] = a_max
                    best_policy = False
        if not best_policy:
            raise Exception(f"Optimal policy not found within {n_iter} iterations")
        return v, q, pi

    def _get_actions(self, s):
        if isinstance(self.actions, dict):
            return {None} if s not in self.actions else self.actions[s]
        elif isinstance(self.actions, set):
            return self.actions
        else: # isinstance(self.actions, types.FunctionType)
            return self.actions(s)

    def _get_transitions(self, s, a, default = None):
        if isinstance(self.transitions, dict):
            return self.transitions[(s, a)] if default is None else self.transitions.get((s, a), default)
        else: # isinstance(self.transitions, types.FunctionType)
            ret_val = self.transitions(s, a)
            return ret_val if ret_val is not None else default
