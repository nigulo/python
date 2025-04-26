import sys
sys.path.append("../")
sys.path.append("../../..")
import numpy as np

from dp import DP
import utils.plot as plot

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

N = 4

if __name__ == '__main__':

    actions = {LEFT, RIGHT, UP, DOWN}
    states = set()
    transitions = {}
    for x in range(N):
        for y in range(N):
            states.add((x, y))
            for a in actions:
                if x == 0 and y == 0:
                    transitions[((x, y), a)] = [((x, y), 0, 0)]
                elif x == 3 and y == 3:
                    transitions[((x, y), a)] = [((x, y), 0, 0)]
                elif a == LEFT:
                    if x == 0:
                        transitions[((x, y), a)] = [((x, y), -1, 1)]
                    else:
                        transitions[((x, y), a)] = [((x-1, y), -1, 1)]
                elif a == RIGHT:
                    if x == N-1:
                        transitions[((x, y), a)] = [((x, y), -1, 1)]
                    else:
                        transitions[((x, y), a)] = [((x+1, y), -1, 1)]
                elif a == DOWN:
                    if y == 0:
                        transitions[((x, y), a)] = [((x, y), -1, 1)]
                    else:
                        transitions[((x, y), a)] = [((x, y-1), -1, 1)]
                else:
                    if y == N-1:
                        transitions[((x, y), a)] = [((x, y), -1, 1)]
                    else:
                        transitions[((x, y), a)] = [((x, y+1), -1, 1)]
                        
                    
    dp = DP(states, actions, transitions)
    v, q, pi = dp.solve(pi_or_v_iter=False)

    plt = plot.plot(nrows=1, ncols=2, size=plot.default_size(200, 200))

    value = np.empty((N, N))
    plt.set_axis_limits([0], limits=[[0, N], [0, N]])

    for x in range(N):
        for y in range(N):
            a = pi.get((x, y), 0)
            value[x, y] = v.get((x, y), 0)
            if a == LEFT: 
                x_arr = x + 0.9
                dx = -0.8
                y_arr = y + 0.5
                dy = 0
            elif a == RIGHT:
                x_arr = x + 0.1
                dx = 0.8
                y_arr = y + 0.5
                dy = 0
            elif a == UP:
                x_arr = x + 0.5
                dx = 0
                y_arr = y + 0.1
                dy = 0.8
            else:
                x_arr = x + 0.5
                dx = 0
                y_arr = y + 0.9
                dy = -0.8
            plt.rectangle(x, y, x+1, y+1, ax_index=[0], facecolor="k", edgecolor="k", linestyle='-', linewidth=1.5)
            plt.arrow(x_arr, y_arr, dx, dy, ax_index=[0], head_width=0.1, facecolor="k", overhang=0.5, length_includes_head=True)
    
    plt.colormap(value.T[::-1,::], [1], show_colorbar=True)
    
    plt.set_axis_title("Policy", [0])
    plt.set_axis_title("Value", [1])
    plt.set_axis_ticks([0], None)
    plt.set_axis_ticks([1], None)
    plt.set_colorbar(ax_index=[1], show_colorbar=True)
    plt.save("small_grid_world.png")
    plt.close()    

