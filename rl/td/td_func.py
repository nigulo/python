import numpy as np
import random
from enum import Enum
from collections import deque

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
    #             b: Callable[[int, int, Callable[int]], Tuple[List[int], List[float]]], 
    #             s0: Callable[[], int], 
    def __init__(self, actions, transitions, d, q, q_grad, b=None, state=None, eps=0.5):
        self.actions = actions
        self.transitions = transitions
        self.w = np.zeros(d)
        self.q = q
        self.q_grad = q_grad
        self.b = b
        self.state = state
        self.eps = eps
        
        self.reset()
                
    def train(self, s0, alpha=0.1, beta=0.1, n_episodes=1000, max_steps=100_000, method=Method.EXPECTED_SARSA, n_steps=1, episodic=True):
        for _ in range(self.episode, self.episode + n_episodes):
            s, a_p = self.next_episode(n_steps, s0=s0)
            for step in range(max_steps):
                s_r_a_p = self.step(step, (s, a_p), alpha, beta, method, n_steps)
                if not s_r_a_p:
                    break
                s, r, a_p = s_r_a_p
            if episodic and step == max_steps - 1:
                print("Maximum number of steps reached")

        return self.w
        
    def next_episode(self, n_steps=1, s0=None):
        self.episode += 1
        return self.init(n_steps, s0)
        

    def init(self, n_steps=1, s0=None):
        self.avg_reward = 0
        self.rewards_buf = deque(maxlen=n_steps)
        self.states_buf = deque(maxlen=n_steps)
        self.actions_buf = deque(maxlen=n_steps)

        if s0:
            s = s0(self.episode)
            a, p = self._get_b_action_prob(s, self.episode)
            return s, (a, p)
    
    def step(self, step, s_a_p, alpha=0.1, beta=0.1, method=Method.EXPECTED_SARSA, n_steps=1):
        self.current_step += 1
        s, (a, p) = s_a_p
        self.states_buf.append(s)
        self.actions_buf.append(a)
        
        s_r = self._random_transition(s, a)
        if s_r:
            s_prime, r_prime = s_r
            self.rewards_buf.append(r_prime)
            a_prime, p_prime = self._get_b_action_prob(s_prime, self.episode)
        s0 = self.states_buf[0]
        a0 = self.actions_buf[0]
        if step >= n_steps - 1 or not s_r:
            if s_r:
                if method == Method.SARSA:
                    q_s_a_prime = self.q(s_prime, a_prime, self.w)
                elif method == Method.EXPECTED_SARSA:
                    q_s_a_prime = 0
                    actions, probs = self._get_pi_actions_probs(s_prime)
                    for i_a, a_prime2 in enumerate(actions):
                        q_s_a_prime += probs[i_a]*self.q(s_prime, a_prime2, self.w)
                else: # Q_LEARNING
                    if self.q2 is None:
                        q_s_a_prime = -np.inf
                        for a_prime2 in self.actions(s_prime):
                            q_s_a_prime = max(q_s_a_prime, self.q(s_prime, a_prime2, self.w))
                    else:
                        q_s_a_prime_max = -np.inf
                        a_max = None
                        for a_prime2 in self.actions(s_prime):
                            q_s_a_prime = self.q(s_prime, a_prime2, self.w)
                            if q_s_a_prime > q_s_a_prime_max:
                                q_s_a_prime_max = q_s_a_prime
                                a_max = a_prime2
                        q_s_a_prime = self.q(s_prime, a_max, self.w)
                delta = self._calc_delta(q_s_a_prime, s0, a0)
                    
            else:
                delta = self._calc_delta(None, s0, a0)

            self.avg_reward += beta*delta
            w = self.w
            self.w += alpha*delta*self.q_grad(s0, a0, w)

        if not s_r:
            #print(None)
            return None

        #print(s_prime, r_prime, a_prime)
        return s_prime, r_prime, (a_prime, p_prime)
    
    def _calc_delta(self, q_s_a_prime, s0, a0):
        delta = np.sum(self.rewards_buf) - self.avg_reward*(len(self.rewards_buf))
        if q_s_a_prime is not None:
            delta += q_s_a_prime - self.q(s0, a0, self.w)
        return delta
                
    def _get_pi_actions_probs(self, s):
        actions = list(self.actions(s))
        q_s_a = [self.q(s, a, self.w) for a in actions]
        q_max_ind = np.argmax(q_s_a)
        
        n_actions = len(actions)
        norm = n_actions - 1
        a_max_prob = 1 - self.eps
        probs = [(1-a_max_prob)/norm]*n_actions
        probs[q_max_ind] = a_max_prob
        return actions, probs
    
    def _get_b_action_prob(self, s, episode):
        r = random.random()
        if self.b is not None:
            a, p = self.b(s, episode)
            ind = np.argmax(np.cumsum(p) >= random.random())
            return a[ind], p[ind]
        
        actions = list(self.actions(s))
        q_s_a = [self.q(s, a, self.w) for a in actions]
        q_max_ind = np.argmax(q_s_a)
        
        if r > self.eps:
            return actions[q_max_ind], 1 - self.eps
        r /= self.eps
        r = int(len(actions)*r)
        #print(s, list(actions), r)
        a = actions[r]
        return a, self.eps
    
    def _random_transition(self, s, a):
        s_r_p = self.transitions(s, a)
        if not s_r_p:
            return None
        s, r, p = list(zip(*s_r_p))
        ind = np.argmax(np.cumsum(p) >= random.random())
        return s[ind], r[ind]

    def get_state(self):
        return self.w, self.episode, self.current_step
    
    def set_state(self, state):
        self.w, self.episode, self.current_step = state
    
    def get_result(self):
        return self.w

    def get_episode(self):
        return self.episode

    def get_current_step(self):
        return self.current_step
    
    def reset(self):
        if self.state:
            self.w, self.episode, self.current_step = self.state
        else:
            self.episode = 0
            self.current_step = 0
