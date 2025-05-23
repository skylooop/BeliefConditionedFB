from gymnasium.envs.registration import register
import numpy as np
from envs.ogbench.generate_maze_maps import parse_maze

# SET TO ABSOLUTE PATH 
maps = np.load("../aux_data/pointmaze-maps.npy", allow_pickle=True)[()]

for i in range(len(maps)):
    register(
        id=f'pointmaze-medium-layout-{i}',
        entry_point='envs.ogbench.make_maze:make_maze_env',
        max_episode_steps=250,
        kwargs=dict(
            loco_env_type='point',
            maze_env_type='maze',
            maze_map=maps[i]
        ),
    )