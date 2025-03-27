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
    
    # Create borders
    maze[0, :] = 1  # up
    maze[-1, :] = 1  # down
    maze[:, 0] = 1  # left
    maze[:, -1] = 1  # right
    
    # Randomly place vertical and horizontal walls
    vertical_wall_pos = np.random.randint(low=3, high=width-3)
    horizontal_wall_pos = np.random.randint(low=3, high=height-3)
    
    # Draw horizontal wall with two doors
    maze[horizontal_wall_pos, :] = 1
    random_horizontal_door1 = np.random.randint(low=1, high=vertical_wall_pos)
    random_horizontal_door2 = np.random.randint(low=vertical_wall_pos + 1, high=width-1)
    maze[horizontal_wall_pos, random_horizontal_door1] = 0
    maze[horizontal_wall_pos, random_horizontal_door2] = 0
    
    # Draw vertical wall with two doors
    maze[:, vertical_wall_pos] = 1
    random_vertical_door1 = np.random.randint(low=1, high=horizontal_wall_pos)
    random_vertical_door2 = np.random.randint(low=horizontal_wall_pos + 1, high=height-1)
    maze[random_vertical_door1, vertical_wall_pos] = 0
    maze[random_vertical_door2, vertical_wall_pos] = 0
    
    return maze
    