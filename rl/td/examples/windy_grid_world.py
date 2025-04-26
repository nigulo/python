import sys
sys.path.append("../")
sys.path.append("../../..")
import numpy as np

from sarsa import Sarsa
import utils.plot as plot

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

NX = len(WIND)
NY = 7

X_START, Y_START = 0, 3
X_GOAL, Y_GOAL = 7, 3


def actions(s):
    x, y = s
    actions = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)}
    if x == 0:
        actions = actions.difference({(-1, -1), (-1, 0), (-1, 1)})
    if y == 0:
        actions = actions.difference({(-1, -1), (0, -1), (1, -1)})
    if x == NX - 1:
        actions = actions.difference({(1, -1), (1, 0), (1, 1)})
    if y == NY - 1:
        actions = actions.difference({(-1, 1), (0, 1), (1, 1)})
    return actions                

def transitions(s, a):
    if s == (X_GOAL, Y_GOAL):
        return []
    x, y = s
    dx, dy = a
    dy += WIND[x]
    return [((x + dx, y + dy), -1, 1)]

if __name__ == '__main__':                        
                    
    sarsa = Sarsa(actions, transitions, (lambda: (X_START, Y_START)))
    q, pi = sarsa.train(n_episodes=1)
    print(q)
    print(pi)

    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(NX*25, NY*25))
    plt.set_axis_limits([0], limits=[[0, NX], [0, NY]])

    for x in range(NX):
        color = 1-WIND[x]/max(WIND)/2
        for y in range(NY):
            plt.rectangle(x, y, x+1, y+1, facecolor=(color, color, color), edgecolor=None, linestyle=None, linewidth=0, fill=True)
            plt.rectangle(x, y, x+1, y+1, facecolor="k", edgecolor="k", linestyle='-', linewidth=1.5)
            if (x, y) == (X_START, Y_START):
                plt.text(x+0.5, y+0.5, "S", size=20.0, ha="center", va="center")
            elif (x, y) == (X_GOAL, Y_GOAL):
                plt.text(x+0.5, y+0.5, "G", size=20.0, ha="center", va="center")

    x, y = X_START, Y_START
    for step in range(100_000):
        if (x, y) == (X_GOAL, Y_GOAL):
            break
        if (x, y) not in pi:
            break
        a = pi[(x, y)]
        [(x_prime, y_prime), _, _] = transitions((x, y), a)
        plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, color='lightblue', linestyle='-', linewidth=1.5)
    
    plt.set_axis_ticks(None)
    plt.save("windy_grid_world.png")
    plt.close()    
