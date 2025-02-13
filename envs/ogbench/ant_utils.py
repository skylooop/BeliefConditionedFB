import gymnasium
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shapely
from shapely import Point, GeometryCollection

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
        polyg = []
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
                    xy = (j * S - torso_x - S / 2, i * S - torso_y - S / 2)
                    rect = patches.Rectangle((j *S - torso_x - S/ 2,
                                            i * S- torso_y - S/ 2),
                                            S,
                                            S, linewidth=1, facecolor='RosyBrown', alpha=1.0)
                    sharply_pol = shapely.Polygon((xy, (xy[0], xy[1] + S), (xy[0]+S, xy[1]+S), (xy[0] +S, xy[1])))
                    polyg.append(sharply_pol)
                    
                    ax.add_patch(rect)
                    
        ax.set_xlim(0 - S /2 + 0.6 * S - torso_x, len(self.unwrapped.maze_map[0]) * S - torso_x - S/2 - S * 0.6)
        ax.set_ylim(0 - S/2 + 0.6 * S - torso_y, len(self.unwrapped.maze_map) * S - torso_y - S/2 - S * 0.6)
        # ax.axis('off')
        return polyg

    def draw_tsne(polygons, sample_obs):
        coll_pol = GeometryCollection(polygons)
        test_obs = []
        for x in np.arange(-2, 9, 0.5):
            for y in np.arange(-2, 9, 0.5):
                if not coll_pol.contains(Point(x, y)):
                    test_obs.append(np.concatenate([[x, y], sample_obs[2:]]))
        test_obs = np.array(test_obs)
        