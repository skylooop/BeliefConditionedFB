from gymnasium.envs.registration import register
import numpy as np
from envs.ogbench.generate_maze_maps import parse_maze

maps = np.load("/home/m_bobrin/ZeroShotRL/aux_data/pointmaze-maps.npy", allow_pickle=True)[()]

register(
    id='pointmaze-medium-layout1',
    entry_point='envs.ogbench.make_maze:make_maze_env',
    max_episode_steps=100,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_map=parse_maze(maps[1])
    ),
)

register(
    id='pointmaze-medium-layout2',
    entry_point='envs.ogbench.make_maze:make_maze_env',
    max_episode_steps=100,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_map=parse_maze(maps[2])
    ),
)

register(
    id='pointmaze-medium-layout3',
    entry_point='envs.ogbench.make_maze:make_maze_env',
    max_episode_steps=100,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_map=parse_maze(maps[3])
    ),
)