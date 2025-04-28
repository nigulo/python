import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../")
sys.path.append("../../..")

from td import Method
from td import TD
import utils.plot as plot

NX = 12
NY = 4

CLIFF = [(x, 0) for x in range(1, NX -1)]
X_START, Y_START = 0, 0
X_GOAL, Y_GOAL = NX - 1, 0

STATE_A = 0
STATE_B = 1
STATE_C = 2
STATE_D = 3

def actions(s):
    if s == STATE_A:
        return [0, 1]
    elif s in {STATE_C, STATE_D}:
        return {None}
    return np.arange(10)

def transitions(s, a):
    if s == STATE_A and a == 0:
        return [(STATE_B, 0, 1)]
    elif s == STATE_A and a == 1:
        return [(STATE_C, 0, 1)]
    elif s == STATE_B:
        return [(STATE_D, np.random.normal(-0.1, 1), 1)]
    else:
        return []

if __name__ == '__main__':                        
    n_epochs = 1000

    n_episodes = np.arange(10, 320, step=20)

    lefts = np.zeros_like(n_episodes)
    double_lefts = np.zeros_like(n_episodes)

    t = tqdm(total=np.sum(n_episodes)*n_epochs*2, desc="Calculating")
    for i, ne in enumerate(n_episodes):
        td = TD(actions, transitions, (lambda _: STATE_A))
        for _ in range(n_epochs):
            q, pi = td.train(n_episodes=ne, method=Method.Q_LEARNING)
            [(s, _, _)] = transitions(STATE_A, pi[STATE_A])
            lefts[i] += (s == STATE_B)
            t.update(ne)

        td = TD(actions, transitions, (lambda _: STATE_A), double=True)
        for _ in range(n_epochs):
            q, pi = td.train(n_episodes=ne, method=Method.Q_LEARNING)
            [(s, _, _)] = transitions(STATE_A, pi[STATE_A])
            double_lefts[i] += (s == STATE_B)
            t.update(ne)
        
    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(250, 150))
    
    plt.plot(n_episodes, lefts/n_epochs*100, "r")
    plt.plot(n_episodes, double_lefts/n_epochs*100, "g")
    plt.set_axis_labels(labels=["Episodes", "% left actions"])
    plt.legend(legends=["Q-learning", "Double Q-learning"])
    
    plt.tight_layout()

    plt.save("double_learning.png")
    plt.close()    
