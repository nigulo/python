import numpy as np
import numpy.random as random

from typing import List, Dict, Set, Tuple, Callable
import types

class Sarsa:
    
    #def __init__(self, 
    #             actions: Callable[int, Set[int]], 
    #             transitions: Callable[[int, int], List[Tuple[int, float, float]]]], 
    #             b: Callable[int, List[Tuple[int, float]]]], 
    #             s0: Callable[[], int], 
    def __init__(self, actions, transitions, s0, state=None, eps=0.1):
        self.actions = actions
        self.transitions = transitions
        self.s0 = s0
        self.state = state
        self.eps = eps
                
    def train(self, gamma=0.9, alpha=0.1, n_episodes=1000, max_steps=100_000):
        q: Dict[Tuple[int, int], float] = {}        
        self.pi: Dict[int, int] = {}

        episode = 0
        if self.state:
            q, self.pi, episode = self.state
            
        pi = self.pi

        for _ in range(episode, episode + n_episodes):
            s = self.s0()
            a = self._get_action(s)                
            for step in range(max_steps):
                s_r = self._random_transition(s, a)
                if not s_r:
                    break
                s_prime, r_prime = s_r
                a_prime = self._get_action(s_prime)
                q_s_a = q.get((s, a), 0)
                q_s_a += alpha*(r_prime + gamma*q.get((s_prime, a_prime), 0) - q_s_a)
                q[(s, a)] = q_s_a
                
                if q_s_a >= q.get((s, pi.get(s, a)), 0):
                    pi[s] = a

                s = s_prime
                a = a_prime
            if step == max_steps:
                print("Maximum number of steps reached")

        self.state = q, pi, n_episodes
        return q, pi
    
    def _pi0(self, s):
        actions = self.actions(s)
        p = 1/len(actions)
        return [(a, p) for a in actions]
    
    def _get_action(self, s):
        if random.random() < self.eps and s in self.pi:
            return self.pi[s]
        a, p = list(zip(*self._pi0(s)))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return a[ind]
    
    def _random_transition(self, s, a):
        s_r_p = self.transitions(s, a)
        if not s_r_p:
            return None
        s, r, p = list(zip(*s_r_p))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return s[ind], r[ind]

    def get_state(self):
        return self.state
    
    def get_result(self):
        q = {}        
        pi = {}
        if self.state:
            q, pi, _ = self.state
        return q, pi