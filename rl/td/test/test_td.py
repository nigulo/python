import numpy as np
import unittest

from td import Method
from td import TD

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

N = 4

def s0(episode):
    return episode % N, (episode // 4) % 4
    
def actions(s):
    x, y = s
    actions = {LEFT, RIGHT, UP, DOWN}
    if x == 0:
        actions.remove(LEFT)
    if y == 0:
        actions.remove(DOWN)
    if x == N - 1:
        actions.remove(RIGHT)
    if y == N - 1:
        actions.remove(UP)
    return actions                
    
def transitions(s, a):
    x, y = s
    if x == 0 and y == 0:
        return []
    elif x == N-1 and y == N-1:
        return []
    elif a == LEFT:
        return [((x-1, y), -1, 1)]
    elif a == RIGHT:
        return [((x+1, y), -1, 1)]
    elif a == DOWN:
        return [((x, y-1), -1, 1)]
    else:
        return [((x, y+1), -1, 1)]

class TestTD(unittest.TestCase):
    
    def test_q_learning(self):                        
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.Q_LEARNING)
                
        self.assert_pi(pi)

    def test_sarsa(self):
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.SARSA)
        
        self.assert_pi(pi)

    def test_expected_sarsa(self):
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.EXPECTED_SARSA)
                
        self.assert_pi(pi)

    def test_q_learning_n_steps_2(self):                        
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.Q_LEARNING, n_steps=2)
                
        self.assert_pi(pi, n_steps=2)

    def test_sarsa_n_steps_2(self):
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.SARSA, n_steps=2)
        
        self.assert_pi(pi, n_steps=2)

    def test_expected_sarsa_n_steps_2(self):
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=2000, method=Method.EXPECTED_SARSA, n_steps=2)
                
        self.assert_pi(pi, n_steps=2)

    def assert_pi(self, pi, n_steps=1):
        expected_keys = set([(x, y) for x in range(N) for y in range(N)])
        self.assertEqual(pi.keys(), expected_keys)

        for s in pi.keys():
            if s == (1, 1):
                self.assertIn(pi[s], {LEFT, DOWN})
            elif s == (2, 2):
                self.assertIn(pi[s], {RIGHT, UP})
            elif s == (0, 3):
                self.assertIn(pi[s], {DOWN, RIGHT})
            elif s == (3, 0):
                self.assertIn(pi[s], {LEFT, UP})
            elif s in [(0, 1), (0, 2)]:
                self.assertEqual(pi[s], DOWN)
            elif s in [(1, 0), (2, 0)]:
                self.assertEqual(pi[s], LEFT)
            elif s in [(3, 1), (3, 2)]:
                self.assertEqual(pi[s], UP)
            elif s in [(1, 3), (2, 3)]:
                self.assertEqual(pi[s], RIGHT)
            #elif s in [(1, 2), (2, 1)]: # all actions are optimal
            #    pass
        