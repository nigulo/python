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
    def __init__(self, actions, transitions, b, s0, state=None, discounting_aware=False):
        self.actions = actions
        self.transitions = transitions
        self.b = b
        self.s0 = s0
        self.state = state
        self.discounting_aware = discounting_aware
        
    def train(self, gamma=0.9, n_episodes=1000):
        discounting_aware = self.discounting_aware
        q: Dict[int, Dict[int, float]] = {}        
        pi: Dict[int, int] = {}
        c: Dict[Tuple[int, int], float] = {}
        wg: Dict[Tuple[int, int], float] = {} # used only when discounting_aware=True

        one_minus_gamma = 1 - gamma
        episode = 0
        if self.state:
            q, pi, c, wg, episode, discounting_aware = self.state

        for e in range(episode, episode + n_episodes):
            states = []
            ps = []
            rs = []
            as_ = []
            s = self.s0(e)
            while True:
                a, p = self._random_action(s)                
                s_r = self._random_transition(s, a)
                if not s_r:
                    break
                states.append(s)
                s, r = s_r
                ps.append(p)
                rs.append(r)
                as_.append(a)
            
            if discounting_aware:
                w_sum = 0
                w_cumsum = 0
                
                wg_sum = 0
                wg_cumsum = 0
            g = 0
            w = 1
            n = len(states) - 1
            for i, s in enumerate(states[::-1]):
                t = n - i
                a_t = as_[t]
                q_s = q.get(s, {})
                q_s_a = q_s.get(a_t, 0)
                c_s_a = c.get((s, a_t), 0)
                if discounting_aware:
                    g += rs[t]
                    w_times_g = w*g

                    c_s_a += one_minus_gamma*w_cumsum + w
                    wg_s_a = wg.get((s, a_t), 0) + one_minus_gamma*wg_cumsum + w_times_g

                    w_cumsum += w_sum + w
                    wg_cumsum += wg_sum + w_times_g
                    w_sum += w
                    wg_sum += w_times_g

                    q_s_a = wg_s_a/c_s_a
                    w *= gamma/ps[t]
                    wg[(s, a_t)] = wg_s_a

                else:
                    g = gamma*g + rs[t]
                    c_s_a += w
                    q_s_a += w/c_s_a*(g - q_s_a)
                    w /= ps[t]
                c[(s, a_t)] = c_s_a
                q_s[a_t] = q_s_a
                q[s] = q_s
                                
                a, w_s = list(zip(*q_s.items()))
                a_max = a[np.argmax(w_s)]
                pi[s] = a_max
                if a_max != a_t:
                    break
        self.state = q, pi, c, wg, e, discounting_aware
        return q, pi
    
    def _random_action(self, s):
        a, p = list(zip(*self.b(s)))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return a[ind], p[ind]
    
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
            q, pi, _, _, _, _ = self.state
        return q, pi