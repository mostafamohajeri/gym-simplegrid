import gym
import gym_simplegrid
import numpy as np

env = gym.make('SimpleGrid-8x8-v0')



nb_states = env.observation_space.n
nb_actions = env.action_space.n
qtable = np.zeros((nb_states, nb_actions))

# Hyperparameters
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = []

# print('Q-table before training:')
# print(qtable)


for _ in range(episodes):
    state, info = env.reset()
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()

        # If random number < epsilon, take a random action
        if rnd < epsilon:
            action = env.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, trunc, info = env.step(action)

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if done:
            outcomes[-1] = "Success"

    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)


print(qtable)

## NOW EXECUTE FROM THE TABLE
state, info = env.reset()
done = False



while not done:
    # Choose the action with the highest value in the current state
    if np.max(qtable[state]) > 0:
        action = np.argmax(qtable[state])

    # If there's no best action (only zeros), take a random one
    else:
        action = env.action_space.sample()

    # Implement this action and move the agent in the desired direction
    new_state, reward, done, trunc, info = env.step(action)
    # Update our current state
    state = new_state
    env.render()

    # If we have a reward, it means that our outcome is a success
    if reward:
        outcomes[-1] = "Success"
        print("Success in", iter)


env.close()