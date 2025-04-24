import numpy as np
import numpy.random as random

from typing import List, Dict, Set, Tuple, Callable
import types

class MC:
    
    #def __init__(self, 
    #             actions: Callable[int, Set[int]], 
    #             transitions: Callable[[int, int], List[Tuple[int, float, float]]]], 
    #             b: Callable[int, List[Tuple[int, float]]]], 
    #             s0: Callable[[], int], 
    def __init__(self, actions, transitions, b, s0, state=None):
        self.actions = actions
        self.transitions = transitions
        self.b = b
        self.s0 = s0
        self.state = state
        
    def train(self, gamma=0.9, n_episodes=1000, eps=1e-7, random_initial_policy=False):
        q: Dict[int, Dict[int, float]] = {}        
        pi: Dict[int, int] = {}
        c: Dict[Tuple[int, int], float] = {}
        episode = 0
        if self.state:
            q, pi, c, episode = self.state

        for _ in range(episode, episode + n_episodes):
            states = []
            ps = []
            rs = []
            as_ = []
            s = self.s0()
            while True:
                a, p = self.random_action(s)                
                s_r = self.random_transition(s, a)
                if not s_r:
                    break
                states.append(s)
                s, r = s_r
                ps.append(p)
                rs.append(r)
                as_.append(a)
            
            g = 0
            w = 1
            n = len(states) - 1
            for i, s in enumerate(states[::-1]):
                t = n - i
                a_t = as_[t]
                g = gamma*g + rs[t]
                c_s_a = c.get((s, a_t), 0) + w
                c[(s, a_t)] = c_s_a
                q_s = q.get(s, {})
                q_s_a = q_s.get(a_t, 0)
                q_s_a += w/c_s_a*(g - q_s_a)
                q_s[a_t] = q_s_a
                q[s] = q_s
                a, w_s = list(zip(*q_s.items()))
                a_max = a[np.argmax(w_s)]
                pi[s] = a_max
                if a_max != a_t:
                    break
                w /= ps[t]
                
        self.state = q, pi, c, n_episodes
        return q, pi
    
    def random_action(self, s):
        a, p = list(zip(*self.b(s)))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return a[ind], p[ind]
    
    def random_transition(self, s, a):
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
            q, pi, _, _ = self.state
        return q, pi