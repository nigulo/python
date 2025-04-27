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
        q, pi = td.train(n_episodes=500, method=Method.Q_LEARNING)
                
        #q_expected = {((0, 1), 0): -1.9, ((0, 1), 1): -2.71, ((0, 1), 2): -2.71, ((0, 1), 3): -1.0, ((1, 2), 0): -2.71, ((1, 2), 1): -2.71, ((1, 2), 2): -2.71, ((1, 2), 3): -2.71, ((2, 1), 0): -2.71, ((2, 1), 1): -2.71, ((2, 1), 2): -2.71, ((2, 1), 3): -2.71, ((0, 0), 0): 0, ((0, 0), 1): 0, ((0, 0), 2): 0, ((0, 0), 3): 0, ((3, 1), 0): -3.439, ((3, 1), 1): -2.71, ((3, 1), 2): -1.9, ((3, 1), 3): -3.439, ((1, 1), 0): -1.9, ((1, 1), 1): -3.439, ((1, 1), 2): -3.439, ((1, 1), 3): -1.9, ((0, 3), 0): -3.439, ((0, 3), 1): -2.71, ((0, 3), 2): -3.439, ((0, 3), 3): -2.71, ((2, 0), 0): -1.9, ((2, 0), 1): -3.439, ((2, 0), 2): -3.439, ((2, 0), 3): -2.71, ((3, 0), 0): -2.71, ((3, 0), 1): -3.439, ((3, 0), 2): -2.71, ((3, 0), 3): -3.439, ((2, 3), 0): -2.71, ((2, 3), 1): -1.0, ((2, 3), 2): -1.9, ((2, 3), 3): -2.71, ((0, 2), 0): -2.71, ((0, 2), 1): -3.439, ((0, 2), 2): -3.439, ((0, 2), 3): -1.9, ((3, 3), 0): 0, ((3, 3), 1): 0, ((3, 3), 2): 0, ((3, 3), 3): 0, ((2, 2), 0): -3.439, ((2, 2), 1): -1.9, ((2, 2), 2): -1.9, ((2, 2), 3): -3.439, ((1, 0), 0): -1.0, ((1, 0), 1): -2.71, ((1, 0), 2): -2.71, ((1, 0), 3): -1.9, ((3, 2), 0): -2.71, ((3, 2), 1): -1.9, ((3, 2), 2): -1.0, ((3, 2), 3): -2.71, ((1, 3), 0): -3.439, ((1, 3), 1): -1.9, ((1, 3), 2): -2.71, ((1, 3), 3): -3.439}
        #self.assertEqual(q, q_expected)
        
        self.assertEqual(pi.keys(), set([(x, y) for x in range(N) for y in range(N)]).difference({(0, 0), (N-1, N-1)}))        
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

    def test_sarsa(self):
                        
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=500, method=Method.SARSA)
                
        #q_expected = {((0, 1), 0): -1.9, ((0, 1), 1): -2.71, ((0, 1), 2): -2.71, ((0, 1), 3): -1.0, ((1, 2), 0): -2.71, ((1, 2), 1): -2.71, ((1, 2), 2): -2.71, ((1, 2), 3): -2.71, ((2, 1), 0): -2.71, ((2, 1), 1): -2.71, ((2, 1), 2): -2.71, ((2, 1), 3): -2.71, ((0, 0), 0): 0, ((0, 0), 1): 0, ((0, 0), 2): 0, ((0, 0), 3): 0, ((3, 1), 0): -3.439, ((3, 1), 1): -2.71, ((3, 1), 2): -1.9, ((3, 1), 3): -3.439, ((1, 1), 0): -1.9, ((1, 1), 1): -3.439, ((1, 1), 2): -3.439, ((1, 1), 3): -1.9, ((0, 3), 0): -3.439, ((0, 3), 1): -2.71, ((0, 3), 2): -3.439, ((0, 3), 3): -2.71, ((2, 0), 0): -1.9, ((2, 0), 1): -3.439, ((2, 0), 2): -3.439, ((2, 0), 3): -2.71, ((3, 0), 0): -2.71, ((3, 0), 1): -3.439, ((3, 0), 2): -2.71, ((3, 0), 3): -3.439, ((2, 3), 0): -2.71, ((2, 3), 1): -1.0, ((2, 3), 2): -1.9, ((2, 3), 3): -2.71, ((0, 2), 0): -2.71, ((0, 2), 1): -3.439, ((0, 2), 2): -3.439, ((0, 2), 3): -1.9, ((3, 3), 0): 0, ((3, 3), 1): 0, ((3, 3), 2): 0, ((3, 3), 3): 0, ((2, 2), 0): -3.439, ((2, 2), 1): -1.9, ((2, 2), 2): -1.9, ((2, 2), 3): -3.439, ((1, 0), 0): -1.0, ((1, 0), 1): -2.71, ((1, 0), 2): -2.71, ((1, 0), 3): -1.9, ((3, 2), 0): -2.71, ((3, 2), 1): -1.9, ((3, 2), 2): -1.0, ((3, 2), 3): -2.71, ((1, 3), 0): -3.439, ((1, 3), 1): -1.9, ((1, 3), 2): -2.71, ((1, 3), 3): -3.439}
        #self.assertEqual(q, q_expected)
        
        self.assertEqual(pi.keys(), set([(x, y) for x in range(N) for y in range(N)]).difference({(0, 0), (N-1, N-1)}))        
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

    def test_expected_sarsa(self):
                        
        td = TD(actions, transitions, s0)
        q, pi = td.train(n_episodes=500, method=Method.EXPECTED_SARSA)
                
        #q_expected = {((0, 1), 0): -1.9, ((0, 1), 1): -2.71, ((0, 1), 2): -2.71, ((0, 1), 3): -1.0, ((1, 2), 0): -2.71, ((1, 2), 1): -2.71, ((1, 2), 2): -2.71, ((1, 2), 3): -2.71, ((2, 1), 0): -2.71, ((2, 1), 1): -2.71, ((2, 1), 2): -2.71, ((2, 1), 3): -2.71, ((0, 0), 0): 0, ((0, 0), 1): 0, ((0, 0), 2): 0, ((0, 0), 3): 0, ((3, 1), 0): -3.439, ((3, 1), 1): -2.71, ((3, 1), 2): -1.9, ((3, 1), 3): -3.439, ((1, 1), 0): -1.9, ((1, 1), 1): -3.439, ((1, 1), 2): -3.439, ((1, 1), 3): -1.9, ((0, 3), 0): -3.439, ((0, 3), 1): -2.71, ((0, 3), 2): -3.439, ((0, 3), 3): -2.71, ((2, 0), 0): -1.9, ((2, 0), 1): -3.439, ((2, 0), 2): -3.439, ((2, 0), 3): -2.71, ((3, 0), 0): -2.71, ((3, 0), 1): -3.439, ((3, 0), 2): -2.71, ((3, 0), 3): -3.439, ((2, 3), 0): -2.71, ((2, 3), 1): -1.0, ((2, 3), 2): -1.9, ((2, 3), 3): -2.71, ((0, 2), 0): -2.71, ((0, 2), 1): -3.439, ((0, 2), 2): -3.439, ((0, 2), 3): -1.9, ((3, 3), 0): 0, ((3, 3), 1): 0, ((3, 3), 2): 0, ((3, 3), 3): 0, ((2, 2), 0): -3.439, ((2, 2), 1): -1.9, ((2, 2), 2): -1.9, ((2, 2), 3): -3.439, ((1, 0), 0): -1.0, ((1, 0), 1): -2.71, ((1, 0), 2): -2.71, ((1, 0), 3): -1.9, ((3, 2), 0): -2.71, ((3, 2), 1): -1.9, ((3, 2), 2): -1.0, ((3, 2), 3): -2.71, ((1, 3), 0): -3.439, ((1, 3), 1): -1.9, ((1, 3), 2): -2.71, ((1, 3), 3): -3.439}
        #self.assertEqual(q, q_expected)

        self.assertEqual(pi.keys(), set([(x, y) for x in range(N) for y in range(N)]).difference({(0, 0), (N-1, N-1)}))        
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
