"""Example for OpenAI Gym."""

import gym
import numpy as np

# 1. Load Environment and Q-table structure
FL_ENV = gym.make('FrozenLake8x8-v0')
Q = np.zeros([FL_ENV.observation_space.n, FL_ENV.action_space.n])

# FL_ENV.obeservation.n, FL_ENV.action_space.n gives number of states and
# action in FL_ENV loaded

# 2. Parameters of Q-leanring
ETA = .628
GAMA = .9
N_EPISODES = 5000
REV_LIST = []  # rewards per episode calculate

# 3. Q-learning Algorithm
for i in range(N_EPISODES):
    print("Episode {ep}".format(ep=i))

    # Reset environment
    current_state = FL_ENV.reset()
    acc_reward = 0
    d = False
    j = 0

    # The Q-Table learning algorithm
    while j < 99:
        print("Q-step: {step}".format(step=j))

        # Render the environment
        FL_ENV.render()

        # Count
        j += 1

        # Choose action from Q table
        a = np.argmax(
            Q[current_state, :] +
            np.random.randn(1, FL_ENV.action_space.n) *
            (1./(i+1))
        )

        # Get new state & reward from environment
        s1, r, d, _ = FL_ENV.step(a)

        # Update Q-Table with new knowledge
        Q[current_state, a] = \
            Q[current_state, a] + \
            ETA * \
            (r + GAMA*np.max(Q[s1, :]) - Q[current_state, a])

        # Count the total reward (or accumulated reward)
        acc_reward += r

        current_state = s1
        if d:  # If it is 1. I.e., termination flag
            break

    REV_LIST.append(acc_reward)
    FL_ENV.render()

print("Reward Sum on all episodes " + str(sum(REV_LIST)/N_EPISODES))
print("Final Values Q-Table")
print(Q)
