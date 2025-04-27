import sys
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


def actions(s):
    x, y = s
    actions = {(-1, 0), (0, -1), (0, 1), (1, 0)}
    if x == 0:
        actions = actions.difference({(-1, 0)})
    if y == 0:
        actions = actions.difference({(0, -1)})
    if x == NX - 1:
        actions = actions.difference({(1, 0)})
    if y == NY - 1:
        actions = actions.difference({(0, 1)})
    return actions                

def transitions(s, a):
    if s == (X_GOAL, Y_GOAL):
        return []
    x, y = s
    dx, dy = a
    x += dx
    y += dy
    if (x, y) in CLIFF:
        return [((X_START, Y_START), -100, 1)]
    return [((x, y), -1, 1)]

if __name__ == '__main__':                        
                    
    td = TD(actions, transitions, (lambda _: (X_START, Y_START)))
    q_sarsa, pi_sarsa = td.train(n_episodes=50, method=Method.SARSA)
    
    td.reset()
    q_q_learning, pi_q_learning = td.train(n_episodes=50, method=Method.Q_LEARNING)

    td.reset()
    q_expected_sarsa, pi_expected_sarsa = td.train(n_episodes=50, method=Method.EXPECTED_SARSA)

    plt = plot.plot(nrows=1, ncols=1, size=plot.default_size(NX*25, NY*25))
    plt.set_axis_limits([0], limits=[[0, NX], [0, NY]])

    for x in range(NX):
        for y in range(NY):
            if (x, y) in CLIFF:
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
        if (x, y) not in pi_sarsa:
            break
        a = pi_sarsa[(x, y)]
        [((x_prime, y_prime), _, _)] = transitions((x, y), a)
        plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, color='blue', linestyle='-', linewidth=1.5)
        x, y = x_prime, y_prime

    x, y = X_START, Y_START
    for _ in range(NX*NY):
        if (x, y) == (X_GOAL, Y_GOAL):
            break
        if (x, y) not in pi_q_learning:
            break
        a = pi_q_learning[(x, y)]
        [((x_prime, y_prime), _, _)] = transitions((x, y), a)
        plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, color='red', linestyle='-', linewidth=1.5)
        x, y = x_prime, y_prime

    x, y = X_START, Y_START
    for _ in range(NX*NY):
        if (x, y) == (X_GOAL, Y_GOAL):
            break
        if (x, y) not in pi_expected_sarsa:
            break
        a = pi_expected_sarsa[(x, y)]
        [((x_prime, y_prime), _, _)] = transitions((x, y), a)
        plt.line(x+0.5, y+0.5, x_prime+0.5, y_prime+0.5, color='green', linestyle=':', linewidth=1.5)
        x, y = x_prime, y_prime
    
    plt.set_axis_ticks(None)
    plt.save("cliff_walking.png")
    plt.close()    
