import numpy as np

def generate_four_room_env(width, height, number_of_space_between_wall=3):
    maze = np.zeros((height, width), dtype=np.int32)
    maze[0, :] = 1 # up
    maze[-1, :] = 1 # down
    maze[:, 0] = 1 # left
    maze[:, -1] = 1 # right
    maze[height // 2, :(width // 2 - number_of_space_between_wall)] = 1 #- number_of_space_between_wall
    maze[height // 2, (width // 2 + number_of_space_between_wall + 1):] = 1
    maze[:, width // 2] = 1
    maze[2, width // 2] = 0
    maze[-3, width // 2] = 0
    maze[height // 2, width // 2 - number_of_space_between_wall + 1:width//2] = 1
    maze[height // 2, width//2 : width // 2 + number_of_space_between_wall] = 1
    return maze
    # rr, cc = rectangle([1, 1], extent=[thick, t_shape[0] + thick])
    # x[rr, cc] = 0
    
    