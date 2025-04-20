import numpy as np
import numpy.random as random
from enum import IntEnum

from typing import List, Dict, Set, Tuple, Callable
import types

class DP:
    
    #def __init__(self, 
    #             states: Set[int], 
    #             actions: Dict[int, Set[int]] | Set[int] | Callable[int, Set[int]], 
    #             probs: Dict[Tuple[int, int], List[Tuple[int, float, float]] | Callable[[int, int], List[Tuple[int, float, float]]]], 
    def __init__(self, 
                 states, 
                 actions, 
                 probs):
        self.states = states
        self.actions = actions
        self.probs = probs
        
    def solve(self, gamma=0.9, n_iter=10000, eps=1e-7, random_policy=False, policy_or_value=True):
        v: Dict[int, float] = {}
        q: Dict[Tuple[int, int], float] = {}        
        pi: Dict[int, int] = {}

        for s in self.states:
            actions = list(self._get_actions(s))
            pi[s] = random.choice(actions) if random_policy else actions[0]

        best_policy = False
        i = 0
        while not best_policy and i < n_iter:
            i += 1
            # value function iteration
            delta = eps
            while delta >= eps:
                delta = 0
                for s in self.states:
                    if policy_or_value:
                        a = pi[s]
                        v_s_max = 0
                        for s_prime, r, p in self._get_probs(s, a, [(None, 0, 0)]):
                            if p > 0:
                                v_s_max += p*(r + gamma*v.get(s_prime, 0))
                    else:
                        a_max = pi[s]
                        v_s_max = -np.inf
                        for a in self._get_actions(s):
                            v_s = 0
                            for s_prime, r, p in self._get_probs(s, a, [(None, 0, 0)]):
                                if p > 0:
                                    v_s += p*(r + gamma*v.get(s_prime, 0))
                            if v_s > v_s_max:
                                v_s_max = v_s
                                a_max = a
                        pi[s] = a_max
                    delta = max(delta, abs(v_s_max - v.get(s, 0)))
                    v[s] = v_s_max
                print(delta)
            
            best_policy = True
            if not policy_or_value:
                break

            # policy iteration
            for s in self.states:
                q_s_a_max = -np.inf
                a_max = pi[s]
                for a in self._get_actions(s):
                    q_s_a = 0
                    for s_prime, r, p in self._get_probs(s, a, [(None, 0, 0)]):
                        if p > 0:
                            q_s_a += p*(r + gamma*v.get(s_prime, 0))
                    q[(s, a)] = q_s_a
                    if q_s_a > q_s_a_max:
                        q_s_a_max = q_s_a
                        a_max = a
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

    def _get_probs(self, s, a, default = None):
        if isinstance(self.probs, dict):
            return self.probs[(s, a)] if default is None else self.probs.get((s, a), default)
        else: # isinstance(self.probs, types.FunctionType)
            ret_val = self.probs(s, a)
            return ret_val if ret_val is not None else default

if __name__ == '__main__':
    
    class State(IntEnum):
        ON = 1
        OFF = 2

    class Action():
        WAIT = 1
        PRESS = 2
        
    states = {State.ON, State.OFF}
    actions = {
        State.ON: {Action.WAIT, Action.PRESS}, 
        State.OFF: {Action.WAIT, Action.PRESS}}
    
    probs = {(State.ON, Action.WAIT): [(State.OFF, 1, 1)],
             (State.ON, Action.PRESS): [(State.ON, -1, 1)],
             (State.OFF, Action.WAIT): [(State.OFF, 0, 0.9), (State.ON, 5, 0.1)],
             (State.OFF, Action.PRESS): [(State.ON, 1, 1)]}
    
    dp = DP(states, actions, probs)
    print(dp.solve())