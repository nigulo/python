import sys
sys.path.append("..")
import numpy as np
import numpy.random as random
import pickle
import os
from tqdm import tqdm
import argparse

from mc import MC

def load(name="state"):
    file = f"{name}.pkl"
    if os.path.isfile(file):
        return pickle.load(open(file, "rb"))
    else:
        return None

def save(data, name="state"):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)
        
class RaceTrack:

    def __init__(self, track):
        self.track = track
        xs, ys = np.where(track == 2)
        self.start = np.column_stack((xs, ys))

    def on_track(self, x, y):
        return 0 <= x < self.track.shape[0] and 0 <= y < self.track.shape[1] and self.track[x, y]
    
    def actions(self, s):
        x, y, v_x, v_y = s
        actions = set()
        for a_x in [-1, 0, 1]:
            v_x1 = v_x + a_x
            if 0 <= v_x1 <= 4:    
                for a_y in [-1, 0, 1]:
                    v_y1 = v_y + a_y
                    if -4 <= v_y1 <= 4:
                        if v_x1 or v_y1:
                            actions.add((a_x, a_y))
        return actions

    def transitions(self, s, a):
        x, y, v_x, v_y = s
        if self.on_track(x, y):
            if self.track[x, y] == 3:
                return []
            a_x, a_y = a
            return [((x + v_x, y + v_y, v_x + a_x, v_y + a_y), -1, 1)]
        x0, y0, v_x0, v_y0 = self.s0()
        return [((x0, y0, v_x0, v_y0), -1, 1)]
    
    def b(self, s):
        actions = self.actions(s)
        p = 1/len(actions)
        return [(a, p) for a in actions]

    def s0(self, _):
        x0, y0 = self.start[random.choice(len(self.start))]
        return x0, y0, 0, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=False, action='store_true')
    args = parser.parse_args()
    
    track = np.genfromtxt("track2.txt", dtype=int, delimiter=1)[::-1]
    
    rt = RaceTrack(track)
    mc = MC(rt.actions, rt.transitions, rt.b, rt.s0, state=load(), discounting_aware=True)
    if args.train:
        for _ in tqdm(range(1), desc="Training"):
            mc.train(gamma=1, n_episodes=100)
            save(mc.get_state())
    q, pi = mc.get_result()
    s = rt.s0(0)
    result = np.array(track)
    while True:
        result[s[0], s[1]] = 4
        if s not in pi:
            break
        transition = rt.transitions(s, pi[s])
        if not transition:
            break
        [(s, _, _)] = transition
        if not rt.on_track(s[0], s[1]):
            break
    print(result)