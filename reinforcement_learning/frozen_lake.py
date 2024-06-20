# Import packages
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
from IPython.display import clear_output

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super(CustomFrozenLakeEnv, self).__init__(**kwargs)

    def step(self, action):
        # Get the next state, reward, done, and info using the parent class's step function
        observation, reward, terminated, truncated, info = super(CustomFrozenLakeEnv, self).step(action)
        # Add penalty if an agnet does not reach the goal
        if reward == 0:  # if the agent doesn't reach the goal
            reward = -0.1  # penalize the agent for each step

        return observation, reward, terminated, truncated, info

    # Create an instance of your custom environment


def random_exploration(env, num_step):
    # Reset the environment to its initial state and get the initial observation (initial state)
    observation, info = env.reset(seed=2023)

    # Simulate the agent's actions for num_step time steps
    rewards = []
    for _ in range(num_step):
        # Choose a random action from the action space
        action = env.action_space.sample()

        # Take the chosen action and observe the resulting state, reward, and termination status
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # If the episode is terminated, reset the environment to the start cell
        if terminated:
            observation, info = env.reset()

        # Display the current state of the environment
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()

    return rewards


env = CustomFrozenLakeEnv(map_name="4x4", is_slippery=False, render_mode="rgb_array")
rewards = random_exploration(env, 20)
print(f"Total reward: {np.round(np.sum(rewards), 2)}")