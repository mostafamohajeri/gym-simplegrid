import gym
import numpy as np
import matplotlib.pyplot as plt
import gym_simplegrid

map = [
        "SEEELLEE",
        "EEEEEEEE",
        "EELWEEEE",
        "EELEEWEE",
        "EEEWEEEE",
        "EWWEEEWE",
        "EWEEWLWE",
        "EEEWEEEG",
    ]

env = gym.make('SimpleGrid-v0', desc = map)

nb_states = env.observation_space.n
nb_actions = env.action_space.n
qtable = np.zeros((nb_states, nb_actions))

# Hyper-parameters
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = np.zeros(episodes)

# print('Q-table before training:')
# print(qtable)


for e in range(episodes):
    state, info = env.reset()
    done = False
    print(state)
    # By default, we consider our outcome to be a failure
    outcomes[e] = 0
    steps = 0
    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done and steps < 200:
        # Generate a random number between 0 and 1
        rnd = np.random.random()
        steps += 1
        # If random number < epsilon, take a random action
        if rnd < epsilon:
            action = env.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(qtable[state[0]])



        # Implement this action and move the agent in the desired direction
        new_state, reward, done, trunc, info = env.step(action)

        # Update Q(s,a)

        qtable[state[0], action] = qtable[state[0], action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state[0]]) - qtable[state[0], action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success

        outcomes[e] = outcomes[e] + 1

    epsilon = max(epsilon - epsilon_decay, 0)



    # Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

# NOW EXECUTE FROM THE TABLE
state, info = env.reset()
done = False

while not done:
    # Choose the action with the highest value in the current state
    if np.max(qtable[state[0]]) > 0:
        action = np.argmax(qtable[state[0]])

    # If there's no best action (only zeros), take a random one
    else:
        action = env.action_space.sample()

    # Implement this action and move the agent in the desired direction
    new_state, reward, done, trunc, info = env.step(action)
    # Update our current state
    state = new_state
    env.render()


env.close()
