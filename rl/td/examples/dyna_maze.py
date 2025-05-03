import sys
sys.path.append("../")
sys.path.append("../../..")
import numpy as np
import random

from dyna import Dyna
from td import TD, Method
import utils.plot as plot


MAZE = np.array([
        [1, 1, 1, 1, 1, 1, 1, 0, 3],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [2, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])[::-1,::]

NX = MAZE.shape[1]
NY = MAZE.shape[0]

Y_START, X_START = np.argwhere(MAZE == 2)[0]
Y_GOAL, X_GOAL = np.argwhere(MAZE == 3)[0]


def actions(s):
    return {(-1, 0), (0, -1), (0, 1), (1, 0)}
    if x == 0:
        actions.remove((-1, 0))
    if y == 0:
        actions.remove((0, -1))
    if x == NX - 1:
        actions.remove((1, 0))
    if y == NY - 1:
        actions.remove((0, 1))
    return actions                

def transitions(s, a):
    if s == (X_GOAL, Y_GOAL):
        return []
    x, y = s
    dx, dy = a
    
    x1 = x + dx
    y1 = y + dy
    
    if x1 < 0 or x1 >= NX:
        return [((x, y), 0, 1)]
    if y1 < 0 or y1 >= NY:
        return [((x, y), 0, 1)]
    if not MAZE[y1, x1]:
        return [((x, y), 0, 1)]

    if MAZE[y1, x1] == 3:
        return [((x1, y1), 1, 1)]
    
    return [((x1, y1), 0, 1)]

def b(s, episode, pi):
    a = list(actions(s))
    n = len(a)
    if s in pi:
        p_pi = 0.1#min(0.9, max(0.1, episode/N_EPISODES))
        p = [(1 - p_pi)/n]*n
        return a + [pi[s]], p + [p_pi]
    return a, [1/n]*n

if __name__ == '__main__':                        
    dyna = Dyna(actions, transitions, b=b)
    q, pi = dyna.plan((lambda _: (X_START, Y_START)), gamma=0.95, n_episodes=6, method=Method.Q_LEARNING)
    #td = TD(actions, transitions, b=b)
    #q, pi = td.train((lambda _: (X_START, Y_START)), gamma=0.95, n_episodes=20, method=Method.Q_LEARNING)

    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(NX*25, NY*25))
    plt.set_axis_limits([0], limits=[[0, NX], [0, NY]])

    for x in range(NX):
        for y in range(NY):
            if not MAZE[y, x]:
                plt.rectangle(x, y, x+1, y+1, facecolor="gray", edgecolor=None, linestyle=None, linewidth=0, fill=True)
            plt.rectangle(x, y, x+1, y+1, facecolor="k", edgecolor="k", linestyle='-', linewidth=1.5)
            if (x, y) == (X_START, Y_START):
                plt.text(x+0.5, y+0.5, "S", size=20.0, ha="center", va="center")
            elif (x, y) == (X_GOAL, Y_GOAL):
                plt.text(x+0.5, y+0.5, "G", size=20.0, ha="center", va="center")

    x, y = X_START, Y_START
    for _ in range(NX*NY):
        if (x, y) == (X_GOAL, Y_GOAL):
            break
        if (x, y) not in pi:
            break
        a = pi[(x, y)]
        [((x_prime, y_prime), _, _)] = transitions((x, y), a)
        plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, color='lightblue', linestyle='-', linewidth=1.5)
        x, y = x_prime, y_prime
    
    plt.set_axis_ticks(None)
    plt.save("dyna_maze.png")
    plt.close()    
