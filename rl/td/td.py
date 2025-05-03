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
    def __init__(self, actions, transitions, s0, b=None, state=None, eps=0.1, double=False):
        self.actions = actions
        self.transitions = transitions
        self.s0 = s0
        self.b = b
        self.state = state
        self.eps = eps
        self.double = double
        
        self.reset()
                
    def train(self, gamma=0.9, alpha=0.1, n_episodes=1000, max_steps=100_000, method=Method.EXPECTED_SARSA, n_steps=1, sigma=lambda _: 1):
        if method not in {Method.SARSA, Method.Q_LEARNING, Method.EXPECTED_SARSA}:
            print("Unsupported method")
            return

        pi = self.pi

        for e in range(self.episode, self.episode + n_episodes):
            rewards_buf = deque(maxlen=n_steps)
            states_buf = deque(maxlen=n_steps)
            actions_buf = deque(maxlen=n_steps)
            rhos_buf = deque(maxlen=n_steps-1)
            sigmas_buf = deque(maxlen=n_steps-1)

            pi_actions_buf = deque(maxlen=n_steps-1)
            pi_probs_buf = deque(maxlen=n_steps-1)

            gammas = gamma**np.arange(n_steps)
            gamma_n = gamma**n_steps


            s = self.s0(e)
            a, p = self._get_b_action_prob(s, e)

            for step in range(max_steps):
                states_buf.append(s)
                actions_buf.append(a)
                sigma_step = sigma(step)
                sigmas_buf.append(sigma_step)
                
                if sigma_step == 1:
                    pi_actions_buf.append(None)
                    pi_probs_buf.append(None)
                else:
                    pi_actions, pi_probs = self._get_pi_actions_probs(s)
                    pi_actions_buf.append(pi_actions)
                    pi_probs_buf.append(pi_probs)
                if sigma_step == 0:
                    rhos_buf.append(None)
                else:
                    rho = (1-self.eps)/p if pi.get(s, a) == a else self.eps/p
                    rhos_buf.append(rho)
                    rhos_prod = np.prod(rhos_buf)

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
                if s_r:
                    s_prime, r_prime = s_r
                    rewards_buf.append(r_prime)
                    a_prime, p_prime = self._get_b_action_prob(s_prime, e)
                all_sigmas_one = np.all(np.array(sigmas_buf) == 1)
                all_sigmas_zero = np.all(np.array(sigmas_buf) == 0)
                if step >= n_steps - 1 or not s_r:
                    if s_r:
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
                        if all_sigmas_one:
                            if rhos_prod:
                                g = np.sum(gammas[:len(rewards_buf)]*rewards_buf) + gamma_n*q_s_a_prime
                        elif all_sigmas_zero:
                            g = self._calc_g_sigma0(q_s_a_prime, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf)
                        else:
                            g = self._calc_g_sigma(q_s_a_prime, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf, sigmas_buf, rhos_buf)
                            
                    else:
                        if all_sigmas_one:
                            g = np.sum(gammas[:len(rewards_buf)-1]*np.asarray(rewards_buf)[:-1])
                        elif all_sigmas_zero:
                            g = self._calc_g_sigma0(None, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf)
                        else:
                            g = self._calc_g_sigma(None, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf, sigmas_buf, rhos_buf)

                    s0 = states_buf[0]
                    a0 = actions_buf[0]
                    q_s_a = q.get((s0, a0), 0)
                    if not all_sigmas_one or rhos_prod:
                        q_s_a_delta = alpha*(g - q_s_a)
                        if all_sigmas_one:
                            q_s_a_delta *= rhos_prod
                        q_s_a += q_s_a_delta    
                        q[(s0, a0)] = q_s_a
                    
                    if s0 not in pi:
                        pi[s0] = a0
                    elif self.q2 is not None:
                        q_s_a_2 = q2.get((s0, a0), 0)
                        if q_s_a + q_s_a_2 > q.get((s0, pi[s0]), 0) + q2.get((s0, pi[s0]), 0):
                            pi[s0] = a0
                    elif q_s_a > q.get((s0, pi[s0]), 0):
                        pi[s0] = a0
                if not s_r:
                    break

                s = s_prime
                a = a_prime
                p = p_prime
            if step == max_steps - 1:
                print("Maximum number of steps reached")

        self.episode = e
        return q, pi
    
    
    def _calc_g_sigma0(self, q_s_a_prime, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf):
        if q_s_a_prime is not None:
            k_start = 2
            g = rewards_buf[-1] + gamma*q_s_a_prime
        else:
            k_start = 3
            g = rewards_buf[-2]
        for k in range(len(rewards_buf) - k_start, 0, -1):
            a_k = actions_buf[k]
            prob_a_k = 0
            g_l = 0
            for l, a_k_l in enumerate(pi_actions_buf[k]):
                if a_k_l == a_k:
                    prob_a_k = pi_probs_buf[k][l]
                    continue
                g_l += pi_probs_buf[k][l]*q2.get((states_buf[k], a_k_l), 0)
            g += rewards_buf[k] + gamma*(g_l + prob_a_k*g)
        return g

    def _calc_g_sigma(self, q_s_a_prime, q2, gamma, rewards_buf, states_buf, actions_buf, pi_actions_buf, pi_probs_buf, sigmas_buf, rhos_buf):
        if q_s_a_prime is not None:
            k_start = 2
            g = rewards_buf[-1] + gamma*q_s_a_prime
        else:
            k_start = 3
            g = rewards_buf[-2]
        for k in range(len(rewards_buf) - k_start, 0, -1):
            a_k = actions_buf[k]
            prob_a_k = 0
            q_a_k = 0
            g_l = 0
            for l, a_k_l in enumerate(pi_actions_buf[k]):
                q_a_k_l = q2.get((states_buf[k], a_k_l), 0)
                if a_k_l == a_k:
                    prob_a_k = pi_probs_buf[k][l]
                    q_a_k = q_a_k_l
                g_l += pi_probs_buf[k][l]*q_a_k_l
            sigma_k = sigmas_buf[k]
            g += rewards_buf[k] + gamma*((sigma_k*rhos_buf[k] + (1-sigma_k)*prob_a_k)*(g - q_a_k) + g_l)
        return g
                
    def _get_pi_actions_probs(self, s):
        actions = list(self.actions(s))
        n_actions = len(actions)
        norm = n_actions
        a_pi_prob = 0
        if s in self.pi:
            a_pi_prob = 1 - self.eps
            norm -= 1
        if norm > 0:
            probs = [(1-a_pi_prob)/norm]*n_actions
        else:
            probs = [0]
        if a_pi_prob > 0:
            probs[actions.index(self.pi[s])] = a_pi_prob
        return actions, probs
    
    def _get_b_action_prob(self, s, episode):
        r = random.random()
        if self.b is not None:
            a, p = self.b(s, episode, self.pi)
            ind = np.argmax(np.cumsum(p) >= random.random())
            return a[ind], p[ind]
        if s in self.pi:
            if r > self.eps:
                return self.pi[s], 1 - self.eps
            r /= self.eps
        actions = self.actions(s)
        r = int(len(actions)*r)
        #print(s, list(actions), r)
        a = list(actions)[r]
        return a, self.eps
    
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
