import gym
import numpy as np
import matplotlib.pyplot as plt
import gym_simplegrid
import gym_simplegrid.envs.mas_simple_grid

# env = gym.make('MASSimpleGrid-8x8-v0')
env = gym_simplegrid.envs.mas_simple_grid.env(map_name="8x8", num_agents=3)

state = env.reset()
done = False

while True:
    # Choose the action with the highest value in the current state
    action_dict = {}
    for agent in env.agents:
        action = env.action_space(agent).sample()
        action_dict[agent] = action

    print(action_dict)
    # Implement this action and move the agent in the desired direction
    new_state, reward, done, trunc, info = env.step(action_dict=action_dict)
    # Update our current state
    state = new_state
    env.render()


env.close()
