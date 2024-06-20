import sys

import numpy as np
import matplotlib.pyplot as plt
from pyboy import PyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from skimage.transform import resize
from IPython.display import clear_output


class RedGymEnv(Env):
    def __init__(self, config):
        super(RedGymEnv, self).__init__()
        # Define action psace
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.valid_actions.extend([
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS
        ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        # Define observation space
        self.output_shape = (144, 160, 1)
        self.output_full_shape = (144, 160, 3)  # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)

        # Define action frequency
        self.act_freq = config['action_freq']

        # Create pyboy object
        head = 'SDL2'
        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        # Initialize the state
        self.init_state = config['init_state']
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

            # Initialize a generator of a game image
        self.screen = self.pyboy.botsupport_manager().screen()

        # Initailize variables to monitor agent's state and reward
        self.agent_stats = []
        self.total_reward = 0

    def render(self):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        return game_pixels_render

    def reset(self):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

            # reset reward value
        self.total_reward = 0
        return self.render(), {}

    def step(self, action):

        # take an aciton
        # press button
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action not to keep taking the action
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

            # render pyBoy image at the last frame of each block
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

        # store the new agent state obtained from the corresponding memory address
        # memory addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
        LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        levels = [self.pyboy.get_memory_value(a) for a in LEVELS_ADDRESSES]
        self.agent_stats.append({
            'x': x_pos, 'y': y_pos, 'levels': levels
        })

        # store the new screen image (i.e. new observation) and reward
        obs_memory = self.render()
        new_reward = levels

        # for simplicity, don't handle terminate or truncated conditions here
        terminated = False  # no max number of step
        truncated = False  # no max number of step

        return obs_memory, new_reward, terminated, truncated, {}

    def close(self):
        self.pyboy.stop()  # terminate pyboy session
        super().close()  # call close function of parent's class




ROM_PATH = "ADD YOUR PATH"
INIT_STATE_FILE_PATH = "ADD YOUR PATH"

env_config = {
            'action_freq': 24, 'init_state': INIT_STATE_PATH,
            'gb_path': ROM_PATH
        }
env = RedGymEnv(env_config)
env.reset()
states = []
rewards = []

try:
    for i in range(30): # run for 30 steps
        random_action = np.random.choice(list(range(len(env.valid_actions))),size=1)[0]
        observation, reward, terminated, truncated, _ = env.step(random_action)
        states.append(observation)
        rewards.append(reward)

        # Display the current state of the environment
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()
finally:
    env.close()

