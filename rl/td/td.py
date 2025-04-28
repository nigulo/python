import numpy as np
import random
from enum import Enum

from typing import List, Dict, Set, Tuple, Callable
import types

class Method(Enum):
    SARSA = 0
    Q_LEARNING = 1
    EXPECTED_SARSA = 2


class TD:
    
    #def __init__(self, 
    #             actions: Callable[int, Set[int]], 
    #             transitions: Callable[[int, int], List[Tuple[int, float, float]]]], 
    #             b: Callable[int, List[Tuple[int, float]]]], 
    #             s0: Callable[[], int], 
    def __init__(self, actions, transitions, s0, b=None, state=None, eps=0.1, double=False):
        self.actions = actions
        self.transitions = transitions
        self.s0 = s0
        self.b = b
        self.state = state
        self.eps = eps
        self.double = double
        
        self.reset()
                
    def train(self, gamma=0.9, alpha=0.1, n_episodes=1000, max_steps=100_000, method=Method.EXPECTED_SARSA):
        if method not in {Method.SARSA, Method.Q_LEARNING, Method.EXPECTED_SARSA}:
            print("Unsupported method")
            return

        pi = self.pi

        for e in range(self.episode, self.episode + n_episodes):
            s = self.s0(e)
            a = self._get_b_action(s)
            for step in range(max_steps):
                if self.q2 is not None:
                    if bool(random.getrandbits(1)):
                        q = self.q
                        q2 = self.q2
                    else:
                        q = self.q2
                        q2 = self.q
                else:
                    q = self.q
                    q2 = self.q
                s_r = self._random_transition(s, a)
                if not s_r:
                    break
                s_prime, r_prime = s_r
                a_prime = self._get_b_action(s_prime)
                q_s_a = q.get((s, a), 0)
                if method == Method.SARSA:
                    q_s_a_prime = q2.get((s_prime, a_prime), 0)
                elif method == Method.EXPECTED_SARSA:
                    q_s_a_prime = 0
                    actions, probs = self._get_pi_actions_probs(s_prime)
                    for i_a, a_prime2 in enumerate(actions):
                        q_s_a_prime += probs[i_a]*q2.get((s_prime, a_prime2), 0)
                else: # Q_LEARNING
                    if self.q2 is None:
                        q_s_a_prime = -np.inf
                        for a_prime2 in self.actions(s_prime):
                            q_s_a_prime = max(q_s_a_prime, q.get((s_prime, a_prime2), 0))
                    else:
                        q_s_a_prime_max = -np.inf
                        a_max = None
                        for a_prime2 in self.actions(s_prime):
                            q_s_a_prime = q.get((s_prime, a_prime2), 0)
                            if q_s_a_prime > q_s_a_prime_max:
                                q_s_a_prime_max = q_s_a_prime
                                a_max = a_prime2
                        q_s_a_prime = q2.get((s_prime, a_max), 0)
                q_s_a += alpha*(r_prime + gamma*q_s_a_prime - q_s_a)
                    
                q[(s, a)] = q_s_a
                
                if s not in pi:
                    pi[s] = a
                elif self.q2 is not None:
                    q_s_a_2 = q2.get((s, a), 0)
                    if q_s_a + q_s_a_2 > q.get((s, pi[s]), 0) + q2.get((s, pi[s]), 0):
                        pi[s] = a
                elif q_s_a > q.get((s, pi[s]), 0):
                    pi[s] = a

                s = s_prime
                a = a_prime
            if step == max_steps - 1:
                print("Maximum number of steps reached")

        self.episode = e
        return q, pi
    
    def _get_pi_actions_probs(self, s):
        actions = list(self.actions(s))
        n_actions = len(actions)
        norm = n_actions
        a_pi_prob = 0
        if s in self.pi:
            a_pi_prob = 1 - self.eps
            norm -= 1
        probs = [(1-a_pi_prob)/norm]*n_actions
        if a_pi_prob > 0:
            probs[actions.index(self.pi[s])] = a_pi_prob
        return actions, probs
    
    def _get_b_action(self, s):
        if self.b is not None:
            return self.b(s)
        r = random.random()
        if s in self.pi:
            if r > self.eps:
                return self.pi[s]
            r /= self.eps
        actions = self.actions(s)
        r = int(len(actions)*r)
        a = list(actions)[r]
        return a 
    
    def _random_transition(self, s, a):
        s_r_p = self.transitions(s, a)
        if not s_r_p:
            return None
        s, r, p = list(zip(*s_r_p))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return s[ind], r[ind]

    def get_state(self):
        return self.q, self.pi, self.episode, self.q2
    
    def get_result(self):
        return self.q, self.pi
    
    def reset(self):
        if self.state:
            self.q, self.pi, self.episode, self.q2 = self.state
        else:
            self.q: Dict[Tuple[int, int], float] = {}        
            self.pi: Dict[int, int] = {}
            self.episode = 0
            self.q2 = None
            if self.double:
                self.q2: Dict[Tuple[int, int], float] = {}
