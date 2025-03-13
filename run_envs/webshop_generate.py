import gym
import sys

sys.path.append('..')
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

if __name__ == '__main__':
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)
    env.reset()

    for _ in range(1):
        action = f'search[table]'
        env.step(action)
        obs = env.observation
        print(len(obs))
