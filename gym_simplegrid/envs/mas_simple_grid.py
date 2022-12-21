from __future__ import annotations

from contextlib import closing
from io import StringIO
from typing import Optional
import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np

from gym_simplegrid.grid import SimpleGrid, Wall, Goal, Start, Lava
from gym_simplegrid.window import Window
from gym.utils import seeding
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAPS = {
    "4x4": ["SEEE", "EWEW", "EEEW", "WEEG"],
    "8x8": [
        "SEEELLEE",
        "EEEEEEEE",
        "EELWEEEE",
        "EELEEWEE",
        "EEEWEEEE",
        "EWWEEEWE",
        "EWEEWLWE",
        "EEEWEEEG",
    ],
}

REWARD_MAP = {
    b'E': 0.0,
    b'S': 0.0,
    b'W': -1.0,
    b'G': 1.0,
    b'L': -5.0
}


def env(render_mode=None, desc: list[str] = None, map_name: str = None, reward_map: dict[bytes, float] = None,
        p_noise: float = None, num_agents: int = 1):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = MASSimpleGridEnv(desc, map_name, reward_map, p_noise, num_agents)
    # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
    #     env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


class MASSimpleGridEnv(AECEnv):
    """
    SimplGrid involves navigating a grid from Start(S) to Goal(G) without colliding with any Wall(W) by walking over
    the Empty(E) cell. Optionally, it is possible to introduce a noise in the environment that makes the agent not always 
    move in the intended direction.


    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * ncols + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x5 map can be calculated as follows: 3 * 5 + 3 = 18.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Rewards
    It is possible to customize the rewards for each state by passing a custom reward map through the argument `reward_map`.

    Default reward schedule is:
    - Reach goal(G): +10
    - Reach wall(W): -1
    - Reach empty(E): 0

    ### Arguments
    ```
    gym.make('SimpleGrid-v0', desc=None, map_name=None, p_noise=None)
    ```

    `desc`: Used to specify custom map for frozen lake. For example,

        desc=["SEEE", "EWEW", "EEEW", "WEEG"].

    `map_name`: ID to use any of the preloaded maps.

        "4x4": ["SEEE", "EWEW", "EEEW", "WEEG"]

        "8x8": [
            "SEEEEEEE",
            "EEEEEEEE",
            "EEEWEEEE",
            "EEEEEWEE",
            "EEEWEEEE",
            "EWWEEEWE",
            "EWEEWEWE",
            "EEEWEEEG",
        ]

    `p_noise`: float. Probability of making a random move different than the desired action.
        It must be a value between 0 and 1.
        If not None, then the desired action is taken with probability 1-p_noise.

        For example, if action is left and p_noise=.3, then:
        - P(move left)=.7
        - P(move up)=.1
        - P(move down)=.1
        - P(move right)=.1

    ### Notes on rendering
    To render properly the environment, remember that the point (x,y) in the desc matrix
    corresponds to the point (y,x) in the rendered matrix.

    This is because the rendering code works on width and height while the computation 
    in the environment works on x and y coordinates 

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, desc: list[str] = None, map_name: str = None, reward_map: dict[bytes, float] = None,
                 p_noise: float = None, num_agents: int = 1):
        """
        Parameters
        ----------
        desc: list[str]
            Custom map for the environment.
        map_name: str
            ID to use any of the preloaded maps.
        reward_map: dict[bytes, float]
            Custom reward map.
        p_noise: float
            Probability of making a random move different than the desired action.
            It must be a value between 0 and 1.
            If not None, then the desired action is taken with probability 1-p_noise.

            For example, if action is left and p_noise=.3, then:
            - P(move left)=.7
            - P(move up)=.1
            - P(move down)=.1
            - P(move right)=.1
        """
        self.p_noise = p_noise
        self.desc = self.__initialise_desc(desc, map_name)
        self.nrow, self.ncol = self.desc.shape

        # Reward
        self.reward_map = self.__initialise_reward_map(reward_map)
        self.reward_range = (min(self.reward_map.values()), max(self.reward_map.values()))

        # Initialise action and state spaces
        self.nA = 4
        self.nS = self.nrow * self.ncol

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_spaces = {agent: spaces.Discrete(self.nS) for agent in self.possible_agents}

        self.action_spaces = {agent: spaces.Discrete(self.nA) for agent in self.possible_agents}

        # Initialise env dynamics
        self.initial_state = {agent: None for agent in self.possible_agents}
        self.initial_state_distrib = self.__get_initial_state_distribution(self.desc)
        self.P = self.__get_env_dynamics()

        # Rendering
        self.window = None
        self.grid = self.__initialise_grid_from_desc(self.desc)
        self.fps = 5

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    @staticmethod
    def __initialise_desc(desc: list[str], map_name: str) -> np.ndarray:
        """
        Initialise the desc matrix.

        Parameters
        ----------
        desc: list[list[str]]
            Custom map for the environment.
        map_name: str
            ID to use any of the preloaded maps.

        Returns
        -------
        desc: np.ndarray
            The desc matrix.

        Examples
        --------
        >>> desc = ["SE", "WG"]
        >>> MASSimpleGridEnv.__initialise_desc(desc, None)
        array([[b'S', b'E'],
               [b'W', b'G']], dtype='|S1')
        """
        if desc is not None:
            return np.asarray(desc, dtype="c")
        if desc is None and map_name is None:
            desc = generate_random_map()
            return np.asarray(desc, dtype="c")
        if desc is None and map_name is not None:
            desc = MAPS[map_name]
            return np.asarray(desc, dtype="c")

    @staticmethod
    def __initialise_grid_from_desc(desc: list[str]) -> SimpleGrid:
        """
        Initialise the grid to be rendered from the desc matrix.

        @NOTE: the point (x,y) in the desc matrix corresponds to the
        point (y,x) in the rendered matrix.

        Parameters
        ----------
        desc: list[list[str]]
            Custom map for the environment.
        
        Returns
        -------
        grid: SimpleGrid
            The grid to be rendered.
        """
        nrow, ncol = desc.shape
        grid = SimpleGrid(width=ncol, height=nrow)
        for row in range(nrow):
            for col in range(ncol):
                letter = desc[row, col]
                if letter == b'G':
                    grid.set(col, row, Goal())
                elif letter == b'W':
                    grid.set(col, row, Wall(color='black'))
                elif letter == b'L':
                    grid.set(col, row, Lava())
                else:
                    grid.set(col, row, None)
        return grid

    @staticmethod
    def __initialise_reward_map(reward_map: dict[bytes, float]) -> dict[bytes, float]:
        if reward_map is None:
            return REWARD_MAP
        else:
            return reward_map

    @staticmethod
    def __get_initial_state_distribution(desc: list[str]) -> np.ndarray:
        """
        Get the initial state distribution.
        
        If desc contains multiple times the letter 'S', then the initial
        state distribution will a uniform on the respective states and the
        initial state radomly sampled from it.     

        Parameters
        ----------
        desc: list[str]
            Custom map for the environment.
        
        Returns:
        --------
        initial_state_distrib: np.ndarray

        Examples
        --------
        >>> desc = ["SES", "WEE", "SEG"]
        >>> MASSimpleGridEnv.__get_initial_state_distribution(desc)
        array([0.33333333, 0.        , 0.33333333, 0.        , 0.        ,
        0.        , 0.33333333, 0.        , 0.        ])
        """
        initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        initial_state_distrib /= initial_state_distrib.sum()
        return initial_state_distrib

    def __to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def __to_next_xy(self, row: int, col: int, a: int) -> tuple[int, int]:
        """
        Compute the next position on the grid when starting at (row, col)
        and taking the action a.

        Remember:
        - 0: LEFT
        - 1: DOWN
        - 2: RIGHT
        - 3: UP
        """
        if a == 0:
            col = max(col - 1, 0)
        elif a == 1:
            row = min(row + 1, self.nrow - 1)
        elif a == 2:
            col = min(col + 1, self.ncol - 1)
        elif a == 3:
            row = max(row - 1, 0)
        return (row, col)

    def __transition(self, row: int, col: int, a: int) -> tuple[int, float, bool]:
        """
        Compute next state, reward and done when starting at (row, col)
        and taking the action action a.
        """

        newrow, newcol = self.__to_next_xy(row, col, a)
        newstate = self.__to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]

        if bytes(newletter) in b"W":
            newstate = self.__to_s(row, col)
            newletter = self.desc[row, col]

        done = bytes(newletter) in b"G"

        reward = self.reward_map[newletter]
        return newstate, reward, done

    def __get_env_dynamics(self):
        """
        Compute the dynamics of the environment.

        For each state-action-pair, the following tuple is computed:
            - the probability of transitioning to the next state
            - the next state
            - the reward
            - the done flag
        """
        nrow, ncol = self.nrow, self.ncol
        nA, nS = self.nA, self.nS

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nrow):
            for col in range(ncol):
                s = self.__to_s(row, col)
                for a in range(nA):
                    li = P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"G":
                        li.append((1.0, s, 0, False))  # @NOTE: is reward=0 correct? Probably the value doesn't matter.
                    elif letter in b"W":
                        li.append((1.0, s, 0, False))
                    else:
                        if self.p_noise:
                            li.append((1 - self.p_noise, *self.__transition(row, col, a)))
                            for b in (a_ for a_ in range(nA) if a_ != a):
                                li.append((self.p_noise / (nA - 1), *self.__transition(row, col, b)))
                        else:
                            li.append((1.0, *self.__transition(row, col, a)))
        return P

    def step(self, action_dict: dict):
        transitions = {}
        obs = {}
        state = {}
        reward = {}
        done = {}

        for (agent, action) in action_dict.items():
            transitions[agent] = self.P[self.state[agent]][action]
            i = categorical_sample([t[0] for t in transitions[agent]], self.np_random())
            p, s, r, d = transitions[agent][i]
            obs[agent] = p
            state[agent] = s
            reward[agent] = r
            done[agent] = d
            self.state[agent] = s
            self.lastaction[agent] = action

        return (state, reward, done, done, done)

    _np_random: Optional[np.random.Generator] = None

    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = True,
            options: Optional[dict] = None,
    ):

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: categorical_sample(self.initial_state_distrib, self.np_random()) for agent in self.agents}
        self.observations = {agent: int(self.state[agent]) for agent in self.agents}
        self.num_moves = 0
        self.lastaction = {agent: None for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        return self.observations

    def render(self, mode="human"):
        if mode == "ansi":
            return self.__render_text(self.desc.tolist())
        elif mode == "human":
            return self.__render_gui_multiple_agents()
        elif mode == "rgb_array":
            return self.__render_rgb_array()
        else:
            raise ValueError(f"Unsupported rendering mode {mode}")

    def __render_gui_multiple_agents(self):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly
        render it, we have to pass (y,x) to the grid.render method.
        """
        # agent_pos = {agent: int(self.state[agent] % self.ncol, self.state[agent] // self.ncol) for agent in
        #              self.agents},
        img = self.grid.render(
            tile_size=32,
            agent_pos={agent: (self.state[agent] % self.ncol, self.state[agent] // self.ncol) for agent in self.agents},
            # agent_pos=(self.state["player_0"] % self.ncol, self.state["player_0"] // self.ncol),
            agent_dir=0
        )
        if not self.window:
            self.window = Window('my_custom_env')
            self.window.show(block=False)
        self.window.show_img(img, self.fps)

    def __render_gui(self):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) to the grid.render method.
        """
        # agent_pos = {agent: int(self.state[agent] % self.ncol, self.state[agent] // self.ncol) for agent in
        #              self.agents},
        img = self.grid.render(
            tile_size=32,
            # agent_pos={agent: (self.state[agent] % self.ncol, self.state[agent] // self.ncol) for agent in self.agents},
            agent_pos=(self.state["player_0"] % self.ncol, self.state["player_0"] // self.ncol),
            agent_dir=0
        )
        if not self.window:
            self.window = Window('my_custom_env')
            self.window.show(block=False)
        self.window.show_img(img, self.fps)

    def __render_rgb_array(self):
        """
        Render the environment to an rgb array.
        """
        img = self.grid.render(
            tile_size=32,
            agent_pos=(self.s % self.ncol, self.s // self.ncol),
            agent_dir=0
        )
        return img

    def __render_text(self, desc):
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window:
            self.window.close()
        return


def generate_random_map(size=8, p=0.8):
    """
    Generates a random valid map (one that has a path from start to goal)
    
    Parameters
    ----------
    size: int 
        Size of each side of the grid
    p: float
        Probability that a tile is empty
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "W":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["E", "W"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]
