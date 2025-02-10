import gymnasium
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MazeVizWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)   
        
    # ======== BELOW is helper stuff for drawing and visualizing ======== #
    def get_starting_boundary(self):
        torso_x, torso_y = self.unwrapped._offset_x, self.unwrapped._offset_y
        S =  self.unwrapped._maze_unit
        return (0 - S / 2 + S - torso_x, 0 - S/2 + S - torso_y), (len(self.unwrapped.maze_map[0]) * S - torso_x - S/2 - S, len(self.unwrapped.maze_map) * S - torso_y - S/2 - S)

    def XY(self, n=20, m=30):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04 * (tr[0] - bl[0]) , tr[0] - 0.04 * (tr[0] - bl[0]), m)
        Y = np.linspace(bl[1] + 0.04 * (tr[1] - bl[1]) , tr[1] - 0.04 * (tr[1] - bl[1]), n)
        
        X,Y = np.meshgrid(X,Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states
    
    def draw(self, ax=None, scale=1.0):
        if not ax: ax = plt.gca()
        torso_x, torso_y = self.unwrapped._offset_x, self.unwrapped._offset_y
        S =  self.unwrapped._maze_unit
        if scale < 1.0:
            S *= 0.965
            torso_x -= 0.7
            torso_y -= 0.95
        for i in range(len(self.unwrapped.maze_map)):
            for j in range(len(self.unwrapped.maze_map[0])):
                struct = self.unwrapped.maze_map[i][j]
                if struct == 1:
                    rect = patches.Rectangle((j *S - torso_x - S/ 2,
                                            i * S- torso_y - S/ 2),
                                            S,
                                            S, linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)

                    ax.add_patch(rect)
        ax.set_xlim(0 - S /2 + 0.6 * S - torso_x, len(self.unwrapped.maze_map[0]) * S - torso_x - S/2 - S * 0.6)
        ax.set_ylim(0 - S/2 + 0.6 * S - torso_y, len(self.unwrapped.maze_map) * S - torso_y - S/2 - S * 0.6)
        ax.axis('off')