CHECKPOINT_DIR = './Mario-Reinforcement-Learning/Train' # Change directory accordingly
LOG_DIR = './Mario-Reinforcement-Learning/Logs'  # Change directory accordingly

import gym
import gym_super_mario_bros
from IPython import display
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
# Import PPO for algos
from stable_baselines3 import PPO, DQN
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import RecordVideo
from time import time


# Create the env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# Preprocess it to upload the actions
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last") # Stack frames
env = VecMonitor(env, "./Train/TestMonitor") # Monitor your progress

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Taken from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param chk_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, save_freq: int, check_freq: int, chk_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.chk_dir = chk_dir
        self.save_path = os.path.join(chk_dir, 'models')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.verbose > 0:
                print(f"Saving current model to {os.path.join(self.chk_dir, 'models')}")
            self.model.save(os.path.join(self.save_path, f'iter_{self.n_calls}'))

        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.chk_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {os.path.join(self.chk_dir, 'best_model')}")
                  self.model.save(os.path.join(self.chk_dir, 'best_model'))

        return True

def startGameRand(env):    
    fin = True
    for step in range(100000): 
        if fin: 
            env.reset()
        state, reward, fin, info = env.step(env.action_space.sample())
        env.render()
    env.close()

def startGameModel(env, model):
    state = env.reset()
    while True: 
        action, _ = model.predict(state)
        state, _, _, _ = env.step(action)
        env.render()

def saveGameRand(env, len = 100000, dir = './videos/'):
    env = RecordVideo(env, dir + str(time()) + '/')
    fin = True
    for step in range(len): 
        if fin: 
            env.reset()
        state, reward, fin, info = env.step(env.action_space.sample())
    env.close()    

def saveGameModel(env, model, len = 100000, dir = './videos/'):
    env = RecordVideo(env, dir + str(time()) + '/')
    fin = True
    for step in range(len): 
        if fin: 
            state = env.reset()
        action, _ = model.predict(state)
        state, _, fin, _ = env.step(action)
    env.close()


callback = SaveOnBestTrainingRewardCallback(save_freq=100000, check_freq=1000,
chk_dir=CHECKPOINT_DIR)


""" 
# ppo1
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=1000000, callback=callback)
"""

""" 
# ppo2
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=128)
"""

""" 
# ppo3
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
"""

"""
# dqn1
model = DQN("CnnPolicy",
  env,
  batch_size=192,
  verbose=1,
  learning_starts=10000,
  learning_rate=5e-3,
  exploration_fraction=0.1,
  exploration_initial_eps=1.0,
  exploration_final_eps=0.1,
  train_freq=8,
  buffer_size=10000,
  tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)
"""

"""
# dqn2
model = DQN("CnnPolicy",
  env,
  batch_size=192,
  verbose=1,
  learning_starts=10000,
  learning_rate=5e-3,
  exploration_fraction=0.1,
  exploration_initial_eps=1.0,
  exploration_final_eps=0.1,
  train_freq=32,
  buffer_size=10000,
  tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)
"""

"""
# dqn3
model = DQN("CnnPolicy",
  env,
  batch_size=192,
  verbose=1,
  learning_starts=10000,
  learning_rate=2e-1,
  exploration_fraction=0.1,
  exploration_initial_eps=1.0,
  exploration_final_eps=0.1,
  train_freq=8,
  buffer_size=10000,
  tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)
"""

model = DQN("CnnPolicy",
  env,
  batch_size=192,
  verbose=1,
  learning_starts=10000,
  learning_rate=2e-1,
  exploration_fraction=0.1,
  exploration_initial_eps=1.0,
  exploration_final_eps=0.1,
  train_freq=8,
  buffer_size=10000,
  tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)

# Load and save your trained model
model = PPO.load("./Train/best_model")
saveGameModel(env, model)
