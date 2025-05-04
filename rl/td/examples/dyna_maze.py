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

MAZE_1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1, 1, 1]])[::-1,::]

MAZE_2 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1, 1, 1]])[::-1,::]

MAZE = MAZE_1
NX = MAZE.shape[1]
NY = MAZE.shape[0]

def x_y_start():
    x, y = np.argwhere(MAZE == 2)[0][::-1]
    return x, y
    #return np.argwhere(MAZE == 2)[0][::-1].tolist()

def x_y_goal():
    x, y = np.argwhere(MAZE == 3)[0][::-1]
    return x, y

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
    if s == x_y_goal():
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
    q, pi = dyna.plan((lambda _: x_y_start()), gamma=0.95, n_episodes=6, method=Method.Q_LEARNING)
    #td = TD(actions, transitions, b=b)
    #q, pi = td.train((lambda _: (X_START, Y_START)), gamma=0.95, n_episodes=20, method=Method.Q_LEARNING)

    plt = plot.plot(nrows=2, ncols=2, size=plot.default_size(NX*25, NY*25), ax_spans={(1, 0): (0, 1)})
    plt.set_axis_limits([0, 0], limits=[[0, NX], [0, NY]])
    plt.set_axis_limits([0, 1], limits=[[0, NX], [0, NY]])

    for x in range(NX):
        for y in range(NY):
            ax = [0, 0]
            for maze in [MAZE_1, MAZE_2]:
                if not maze[y, x]:
                    plt.rectangle(x, y, x+1, y+1, ax_index=ax, facecolor="gray", edgecolor=None, linestyle=None, linewidth=0, fill=True)
                plt.rectangle(x, y, x+1, y+1, ax_index=ax, facecolor="k", edgecolor="k", linestyle='-', linewidth=1.5)
                if (x, y) == x_y_start():
                    plt.text(x+0.5, y+0.5, "S", ax_index=ax, size=20.0, ha="center", va="center")
                elif (x, y) == x_y_goal():
                    plt.text(x+0.5, y+0.5, "G", ax_index=ax, size=20.0, ha="center", va="center")
                ax = [0, 1]
    
    dyna = Dyna(actions, transitions, b=b)
    rewards = []
    time_steps = []

    dyna_plus = Dyna(actions, transitions, b=b, kappa=1e-3)
    rewards_plus = []
    time_steps_plus = []
    
    num_steps = 200
    ax = [0, 0]
    for i in range(num_steps):
        if i == num_steps//2:
            MAZE = MAZE_2
            ax = [0, 1]
        for d, rs, ts, draw in [(dyna, rewards, time_steps, False), (dyna_plus, rewards_plus, time_steps_plus, True)]:
            q, pi = d.plan((lambda _: x_y_start()), gamma=0.95, n_episodes=1, method=Method.Q_LEARNING)
    
            x, y = x_y_start()
            for t in range(NX*NY):
                if (x, y) == x_y_goal():
                    break
                if (x, y) not in pi:
                    break
                a = pi[(x, y)]
                [((x_prime, y_prime), r, _)] = transitions((x, y), a)
                if draw and (i == num_steps//2 - 1 or i == num_steps - 1):
                    plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, ax_index=ax, color='lightblue', linestyle='-', linewidth=1.5)
                x, y = x_prime, y_prime
            rs.append(r)
            ts.append(t)
    
    plt.plot(np.cumsum(time_steps), np.cumsum(rewards), "b", ax_index=[1, 0])
    plt.plot(np.cumsum(time_steps_plus), np.cumsum(rewards_plus), "r", ax_index=[1, 0])
    plt.legend(ax_index=[1, 0], legends=["Dyna-Q", "Dyna-Q+"], loc='lower right')
    
    plt.set_axis_ticks(ax_index=[0, 0], ticks=None)
    plt.set_axis_ticks(ax_index=[0, 1], ticks=None)
    plt.set_axis_labels(ax_index=[1, 0], labels=["Time steps", "Cumulative reward"])
    plt.save("dyna_maze.png")
    plt.close()    
