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
    
def fourrooms_random_layouts(width, height):
    maze = np.zeros((height, width), dtype=np.int32)
    # create borders
    maze[0, :] = 1 # up
    maze[-1, :] = 1 # down
    maze[:, 0] = 1 # left
    maze[:, -1] = 1 # right
    
    vertical_wall_pos = np.random.randint(low=3, high=height-5)
    horizontal_wall_pos = np.random.randint(low=3, high=width-4)
    
    maze[horizontal_wall_pos, :] = 1
    random_horizontal_door = np.random.randint(low=1, high=width//2 - 1)
    maze[horizontal_wall_pos, random_horizontal_door] = 0
    
    random_horizontal_door2 = np.random.randint(low=width//2 + 1, high=width-1)
    maze[horizontal_wall_pos, random_horizontal_door2] = 0
    
    maze[:, vertical_wall_pos] = 1
    random_vertical_door = np.random.randint(low=1, high=height//2 -1)
    maze[random_vertical_door, vertical_wall_pos] = 0
    
    random_vertical_door2 = np.random.randint(low=height//2 + 1, high=height-1)
    maze[random_vertical_door2, vertical_wall_pos] = 0
    
    
    return maze
    
    