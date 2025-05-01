import sys
sys.path.append("..")
sys.path.append("../../..")

import numpy as np
import numpy.random as random
import pickle
import os
from tqdm import tqdm
import argparse

from mc import MC
import utils.plot as plot

N_EPISODES = 1000

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

    def finish(self, x, y):
        return self.track[x, y] == 3
    
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
        x0, y0, v_x0, v_y0 = self.s0(None)
        return [((x0, y0, v_x0, v_y0), -1, 1)]
    
    def b(self, s, episode, pi):
        a = list(self.actions(s))
        n = len(a)
        if s in pi:
            p_pi = min(0.9, max(0.1, episode/N_EPISODES))
            p = [(1 - p_pi)/n]*n
            return a + [pi[s]], p + [p_pi]
        return a, [1/n]*n
        
        #a = list(self.actions(s))
        #n = len(a)
        #return a, [1/n]*n

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
        for _ in tqdm(range(N_EPISODES//10), desc="Training"):
            mc.train(gamma=1, n_episodes=10)
            save(mc.get_state())
    q, pi = mc.get_result()

    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(track.shape[1]*5, track.shape[0]*5))
    plt.set_axis_limits(limits=[[0, track.shape[1]], [0, track.shape[0]]])

    for x in range(track.shape[1]):
        for y in range(track.shape[0]):
            if track[y, x] == 1:
                continue
            elif track[y, x] == 2:
                color = (0, 0, 1)
            elif track[y, x] == 3:
                color = (0, 1, 0)
            else:
                color = (0.5, 0.5, 0.5)
            plt.rectangle(x, y, x+1, y+1, facecolor=color, edgecolor=color, linestyle='-', linewidth=1.5, fill=True)

    s = rt.s0(0)
    x, y = s[0], s[1]
    for _ in range(np.product(track.shape)):
        if s not in pi:
            break
        transition = rt.transitions(s, pi[s])
        if not transition:
            break
        [(s_prime, _, _)] = transition
        x_prime, y_prime = s_prime[0], s_prime[1]
        if not rt.on_track(x_prime, y_prime):
            break
        plt.line(y+0.5, x+0.5, y_prime+0.5, x_prime+0.5, color=(1, 0, 0), linestyle='-', linewidth=1.5)

        s = s_prime
        x, y = x_prime, y_prime
        if rt.finish(x, y):
            break
    
    plt.set_axis_ticks(None)
    plt.save("race_track.png")
    plt.close()    
    