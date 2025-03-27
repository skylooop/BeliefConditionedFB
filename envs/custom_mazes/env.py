from abc import ABC
from abc import abstractmethod
from typing import *
import numpy as np
import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt

class BaseEnv(gym.Env, ABC):
    metadata = {
        'render.modes': ['rgb_array']
    }
    
    def __init__(self, max_width:int=500):
        self.reward_range = (-float('inf'), float('inf'))
        self.max_width = max_width
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int]=None, options: Optional[int]=None):
        super().reset(seed=seed)
    
    @abstractmethod
    def get_image(self):
        pass
    
    def render(self, ax=None, return_img=False):
        img = self.get_image()
        grid_kwargs = {'color': (220 / 255, 220 / 255, 220 / 255, 0.5)}
        img = np.asarray(img).astype(np.uint8)
        if return_img:
            img = Image.fromarray(img) # upscale
            img = img.resize((200, 200), Image.NEAREST)
            return np.asarray(img)
        
        if ax is None:
            ax = plt.gca()
        ax.imshow(img, interpolation=None)
        ax.grid(0)
        ax.set_xticks([])
        ax.set_yticks([])
        h, w = img.shape[:2]
        for y in range(h - 1):
            ax.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], **grid_kwargs)
        for x in range(w - 1):
            ax.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], **grid_kwargs)