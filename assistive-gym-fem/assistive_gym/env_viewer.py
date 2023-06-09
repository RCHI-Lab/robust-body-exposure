import gym, sys, argparse
import numpy as np
from .learn import make_env
# import assistive_gym

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)
    all_reward = []
    # while True:
    for i in range(100):
        done = False
        env.render()
        observation = env.reset()

        action = sample_action(env, coop)
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))

        elif 'BeddingManipulationSphere-v1' in env_name:
            action = np.array([0.3, 0.5, 0, 0])
        elif 'RemoveContactSphere-v1' in env_name:
            action = np.array([0.3, 0.45])
        else:
            pass
            # print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not done:
            observation, reward, done, info = env.step(sample_action(env, coop))
            # action = np.array([ 0.15795245,  0.82185944,  0.00900022 ,-0.59316433])
            # observation, reward, done, info = env.step(action)
            if coop:
                done = done['__all__']
        print(f"Trial {i}: Reward = {reward:.2f}")
        all_reward.append(reward)

    print("Mean Reward:", np.mean(all_reward))
    print("Reward Std:", np.std(all_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)
