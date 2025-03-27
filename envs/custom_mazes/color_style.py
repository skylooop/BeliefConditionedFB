from dataclasses import dataclass

@dataclass
class DeepMindColor:
    obstacle = (100, 100, 100)
    free = (255, 255, 255)
    agent = (51, 153, 255)
    goal = (0, 255, 0)
    button = (102, 0, 204)
    interruption = (255, 0, 255)
    box = (0, 102, 102)
    lava = (255, 0, 0)
    water = (0, 0, 255)
