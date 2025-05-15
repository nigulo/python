import numpy as np
import random
from enum import Enum
import heapq

from td import TD, Method

class Dyna:
    
    def __init__(self, actions, transitions, b=None, state=None, eps=0.1, double=False, kappa=0):
        self.td_learn = TD(actions=actions, transitions=transitions, b=lambda  s, episode, pi: self._b(b, s, episode, pi), state=state[0] if state else None, eps=eps, double=double)
        self.model = {}
        if state:
            self.model = state[1]
        self.td_plan = TD(actions=actions, transitions=self._model_transitions, b=b, state=state[0] if state else None, eps=eps, double=double)
        self.kappa = kappa
        
    def plan(self, s0, gamma=0.9, alpha=0.1, n_episodes=100, max_steps=100_000, method=Method.EXPECTED_SARSA, n_steps=1, sigma=lambda _: 1, theta=np.inf, n_plan_steps=100):
        p_queue = []
        p_queue_keys = set()
        for _ in range(self.td_learn.get_episode(), self.td_learn.get_episode() + n_episodes):
            s, a_p = self.td_learn.next_episode(gamma, n_steps, s0=s0)
            for step in range(max_steps):
                a, _ = a_p
                s_r_a_p = self.td_learn.step(step, (s, a_p), gamma, alpha, method, n_steps, sigma)
                if not s_r_a_p:
                    break
                s_prime, r, a_p = s_r_a_p

                self.model[(s, a)] = [(s_prime, r, 1, self.td_learn.get_episode())]
                
                if theta < np.inf:
                    q, _ = self.td_learn.get_result()
                    q_s_a = q[(s, a)]
                    if (s, a) not in p_queue_keys and q_s_a >= theta:
                        heapq.heappush(p_queue, (-q_s_a, (s, a)))
                        p_queue_keys.add((s, a))
                s = s_prime
            self.td_plan.set_state(self.td_learn.get_state())
            if step == max_steps - 1:
                print("Maximum number of steps reached")
                
            if theta == np.inf:
                for i in range(n_plan_steps):
                    s, a = random.choice(list(self.model.keys()))
                    if (s, a) not in p_queue_keys:
                        heapq.heappush(p_queue, (i, (s, a)))
                        p_queue_keys.add((s, a))
                    
            while len(p_queue) > 0:
                _, (s, a) = heapq.heappop(p_queue)
                self.td_plan.init(gamma, n_steps=1)
                self.td_plan.step(step=0, s_a_p=(s, (a, 1)), gamma=gamma, alpha=alpha, method=method, n_steps=1, sigma=sigma)
                if theta == np.inf:
                    continue
                for (s_, a_) in self.model.keys():
                    [(s_prime_, r_, _, _)] = self.model[(s_, a_)]
                    if s_prime_ == s:
                        self.td_plan.init(gamma, n_steps=1)
                        self.td_plan.step(step=0, s_a_p=(s_, (a_, 1)), gamma=gamma, alpha=alpha, method=method, n_steps=1, sigma=sigma)
                        q, _ = self.td_plan.get_result()
                        q_s_a = q[(s_, a_)]
                        if (s_, a_) not in p_queue_keys and q_s_a >= theta:
                            heapq.heappush(p_queue, (-q_s_a, (s_, a_)))
                            p_queue_keys.add((s_, a_))
            p_queue_keys = set()
            self.td_learn.set_state(self.td_plan.get_state())

        return self.td_learn.get_result()
    
    '''
    def _transitions(self, s, a):
        transition = self.transitions(s, a)
        if not transition:
            return []
        [(s_prime, r, p)] = transition
        if not self.kappa:
            return [(s_prime, r, p)]
        dr = 0
        if (s, a) in self.model:
            [(_, _, _, last_episode)] = self.model[(s, a)]
            episode = self.td_learn.get_episode()
            dr = self.kappa*np.sqrt(episode - last_episode)
        return [(s_prime, r + dr, p)]
    '''
    
    def _model_transitions(self, s, a):
        transition = self.model[(s, a)]
        if not transition:
            return []
        [(s_prime, r, p, last_episode)] = transition
        if True:#not self.kappa:
            return [(s_prime, r, p)]        
        episode = self.td_learn.get_episode()
        dr = 0#self.kappa*np.sqrt(episode - last_episode)
        return [(s_prime, r + dr, p)]

    '''
    def _update_last_visits(self, s, a):
        transition = self.model[(s, a)]
        if not transition:
            return
        [(s_prime, r, p, _)] = transition
        self.model[(s, a)] = [(s_prime, r, p, self.td_plan.get_episode())]
    '''

    def _b(self, b, s, episode, pi):
        actions, probs = b(s, episode, pi)
        if not self.kappa:
            return actions, probs
        norm = 0
        for i, a in enumerate(actions):
            if (s, a) in self.model:
                transition = self.model[(s, a)]
                if not transition:
                    return []
                [(_, _, _, last_episode)] = transition
                episode = self.td_learn.get_episode()
                scale = 1 + self.kappa*np.sqrt(episode - last_episode)
                norm += scale*probs[i]
                probs[i] *= scale
        return actions, probs if not norm else np.asarray(probs)/norm

    def get_state(self):
        return self.td_learn.get_state(), self.model
    
    def get_step(self):
        return self.td_learn.get_current_step()
    