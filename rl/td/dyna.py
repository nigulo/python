import numpy as np
import random
from enum import Enum
from collections import deque

from td import TD, Method

class Dyna:
    
    def __init__(self, actions, transitions, b=None, state=None, eps=0.1, double=False, kappa=0):
        self.td_learn = TD(actions=actions, transitions=transitions, b=b, state=state[0] if state else None, eps=eps, double=double)
        self.model = {}
        if state:
            self.model = state[1]
        self.td_plan = TD(actions=actions, transitions=self._model_transitions, b=b, state=state[0] if state else None, eps=eps, double=double)

        #self.transitions = transitions
        self.kappa = kappa
        
    def plan(self, s0, gamma=0.9, alpha=0.1, n_episodes=100, max_steps=100_000, method=Method.EXPECTED_SARSA, n_steps=1, sigma=lambda _: 1, n_plan_steps=50):
        for e in range(self.td_learn.get_episode(), self.td_learn.get_episode() + n_episodes):
            s, a_p = self.td_learn.start_episode(e, gamma, n_steps, s0=s0)
            for step in range(max_steps):
                a, _ = a_p
                s_r_a_p = self.td_learn.step(e, step, (s, a_p), gamma, alpha, method, n_steps, sigma)
                if not s_r_a_p:
                    break
                s_prime, r, a_p = s_r_a_p

                self.model[(s, a)] = [(s_prime, r, 1, self.td_learn.get_current_step())]
                s = s_prime
            self.td_plan.set_state(self.td_learn.get_state())
            if step == max_steps - 1:
                print("Maximum number of steps reached")
            for _ in range(n_plan_steps):
                s, a = random.choice(list(self.model.keys()))
                a_p = (a, 1)
                self.td_plan.start_episode(e, gamma, n_steps)
                for step in range(n_steps):
                    self._update_model(s, a)
                    s_r_a_p = self.td_plan.step(e, step, (s, a_p), gamma, alpha, method, n_steps, sigma)
                    if not s_r_a_p:
                        break
                    s, r, a_p = s_r_a_p
                    a, _ = a_p
            self.td_learn.set_state(self.td_plan.get_state())

        return self.td_learn.get_result()
    
    #def _transitions(self, s, a):
    #    transition = self.transitions(s, a)
    #    if not transition:
    #        return []
    #    [(s_prime, r, p)] = transition
    #    if not self.kappa:
    #        return [(s_prime, r, p)]
    #    current_step = self.td_learn.get_current_step()
    #    last_step = -0.1/self.kappa
    #    if (s, a) in self.model:
    #        [(_, _, _, last_step)] = self.model[(s, a)]
    #    return [(s_prime, r + self.kappa*np.sqrt(current_step - last_step), p)]
    
    def _model_transitions(self, s, a):
        transition = self.model[(s, a)]
        if not transition:
            return []
        [(s_prime, r, p, last_step)] = transition
        if not self.kappa:
            return [(s_prime, r, p)]        
        current_step = self.td_plan.get_current_step()
        dr = self.kappa*np.sqrt(current_step - last_step)
        return [(s_prime, r + dr, p)]

    def _update_model(self, s, a):
        transition = self.model[(s, a)]
        if not transition:
            return
        [(s_prime, r, p, _)] = transition
        self.model[(s, a)] = [(s_prime, r, p, self.td_plan.get_current_step())]

    def get_state(self):
        return self.td_learn.get_state(), self.model
    