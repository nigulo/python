import numpy as np
import unittest

from mc import MC

STATE_1 = 1
STATE_2 = 2
STATE_3 = 3

ACTION_1 = 1
ACTION_2 = 2

def actions(s):
    return {ACTION_1} if s == STATE_1 else {ACTION_1, ACTION_2}

def transitions(s, a, deterministic=True):
    if s == STATE_1:
        if deterministic:
            return [(STATE_2, -1, 1)]
        else:
            return [(STATE_1, -1, 0.5), (STATE_2, -1, 0.5)]
    elif s == STATE_2:
        return [(STATE_1, -1, 1)] if a == ACTION_1 else [(STATE_3, -1, 1)]
    else:
        return []

def b(s, episode, pi):
    a = list(actions(s))
    n = len(a)
    return a, [1/n]*n

def s0(_):
    return STATE_1

class TestMC(unittest.TestCase):
    
    def test(self):
        mc = MC(actions, transitions, b, s0)
        res = mc.train(gamma=1, n_episodes=100)
        
        self.assertEqual(res, mc.get_result())
        
        q, pi = res
        
        q_expected = {1: {1: -2.0}, 2: {1: -3.0, 2: -1.0}}
        pi_expected = {2: 2, 1: 1}
        
        self.assertEqual(q, q_expected)
        self.assertEqual(pi, pi_expected)
        
    def test_nondeterministic_transition(self):
        mc = MC(actions, lambda s, a: transitions(s, a, deterministic=False), b, s0)
        q, pi = mc.train(gamma=1, n_episodes=1000)
        
        q_expected = {2: {2: -1.0, 1: -4}, 1: {1: -3}}
        pi_expected = {2: 2, 1: 1}
        
        self.assertEqual(pi, pi_expected)
        self.assertEqual(q.keys(), q_expected.keys())
        
        for s in q.keys():
            as_ = q[s]
            as_expected = q_expected[s]
            self.assertEqual(as_.keys(), as_expected.keys())
            for a in as_.keys():
                np.testing.assert_almost_equal(as_[a], as_expected[a], 1)
        
    def test_discounting_aware_equals_discounting_unaware_with_unit_gamma(self):
        mc = MC(actions, transitions, b, s0, discounting_aware=False)
        q, pi = mc.train(gamma=1, n_episodes=100)
        mc = MC(actions, transitions, b, s0, discounting_aware=True)
        q_discounting_aware, pi_discounting_aware = mc.train(gamma=1, n_episodes=100)
        
        self.assertEqual(q, q_discounting_aware)
        self.assertEqual(pi, pi_discounting_aware)

    def test_with_gamma(self):
        mc = MC(actions, transitions, b, s0)
        q, pi = mc.train(gamma=0.9, n_episodes=100)
        
        q_expected = {1: {1: -1.9}, 2: {1: -2.71, 2: -1.0}}
        pi_expected = {2: 2, 1: 1}
        
        self.assertEqual(pi, pi_expected)
        self.assertEqual(q, q_expected)

    def test_disounting_aware_with_gamma(self):
        mc = MC(actions, transitions, b, s0, discounting_aware=True)
        q, pi = mc.train(gamma=0.9, n_episodes=100)

        q_expected = {1: {1: -1.94737}, 2: {1: -2.71, 2: -1.0}}
        pi_expected = {2: 2, 1: 1}
        
        self.assertEqual(pi, pi_expected)
        self.assertEqual(q.keys(), q_expected.keys())

        for s in q.keys():
            as_ = q[s]
            as_expected = q_expected[s]
            self.assertEqual(as_.keys(), as_expected.keys())
            for a in as_.keys():
                np.testing.assert_almost_equal(as_[a], as_expected[a], 5)
        
if __name__ == '__main__':
    unittest.main()