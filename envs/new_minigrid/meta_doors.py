from envs.new_minigrid.minigrid import *
from envs.new_minigrid.register import register
from envs.new_minigrid.minigrid import WorldObj, OBJECT_TO_IDX

class GenieEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        name,
        size=8,
        agent_start_pos=(1, 4),
        max_steps=200,
        agent_start_dir=0,
        agent_start_random=True,
    ):
        self.name = name
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_start_random = agent_start_random
        self.has_aux_info = True
        self.has_consulted_genie = False
        self.most_recent_consult_step = -1
        self.max_steps = max_steps
        self.size = size

        self.genie_location = (1, 1)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def make_genie(self, goal_box_idx):
        g = Genie(color='blue', item_location=goal_box_idx)
        g.set_env(self)
        return g

    def get_genie_trigger_locs(self):
        """Walking by these coordinates talks to the genie"""
        return [(1, 2), (2, 1)]

    def generate_agent_start_loc(self):
        possible_locations = []
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.grid.get(i, j) is None and (i,j) not in self.get_genie_trigger_locs():
                    possible_locations.append((i, j))
        idx = self._rand_int(low=0, high=len(possible_locations))
        return possible_locations[idx]

    def get_genie(self):
        return self.grid.get(*self.genie_location)

    def aux_info_dim(self):
        return self.num_boxes

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # goal_box_idx = 0
        goal_box_idx = self._rand_int(low=0, high=self.num_boxes)
        locations = [(width-2, 1), (width-2, height-2), (1,height-2)]
        for i in range(self.num_boxes):
            if i == goal_box_idx:
                box = ImmovableBox(color='green', contains=Goal())
            else:
                box = ImmovableBox(color='green', contains=None)
            self.put_obj(box, locations[i][0], locations[i][1])

        #Change item_location to a cooordinate
        # item_location = locations[goal_box_idx][1] * width + locations[goal_box_idx][0]
        self.put_obj(self.make_genie(goal_box_idx), self.genie_location[0], self.genie_location[1])
        #(coodridnate, colour, integer)

        # Place the agent
        if self.agent_start_pos is not None:
            if self.agent_start_random:
                self.agent_pos = self.generate_agent_start_loc()
                self.agent_dir = self._rand_int(low=0, high=4)
            else:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
        else:
            raise NotImplementedError

        self.mission = "Hello"
        self.item_location = goal_box_idx

        self.has_consulted_genie = False
        self.most_recent_consult_step = -1

        # Used to mask observation
        self.mask_fn = None

    def getItemLocation(self):
        return self.item_location