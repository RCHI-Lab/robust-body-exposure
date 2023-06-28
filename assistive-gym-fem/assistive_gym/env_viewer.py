import gym, sys, argparse
import numpy as np
from .learn import make_env
import os.path 
import pathlib
import time
import pybullet as p
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
    env = make_env(env_name, coop=True, seed=0) if coop else gym.make(env_name)
    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)
    all_reward = []
    #actions = list()
    num_rollouts = 15
    np.random.seed(0)
    actions = np.random.uniform(-1, 1, size=(num_rollouts, 4))

    # while True:
    for i in range(num_rollouts):
        done = False
        env.render()
        observation = env.reset()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        env.set_iteration(i)
        #action = sample_action(env, coop)
        #actions.append(action)
        action = actions[i]
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        elif 'BeddingManipulationSphere-v1' in env_name:
            action = np.array([0.3, 0.5, 0, 0])
        elif 'RemoveContactSphere-v1' in env_name:
            action = np.array([0.3, 0.45])
        else:
            pass
            # print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        #Perform same action every time
        action = np.array([0, 0, -.3, -.3])
        #action = np.array([.3, .5, 0, 0])
 
        while not done:
            observation, reward, done, info = env.step(action)
            # action = np.array([ 0.15795245,  0.82185944,  0.00900022 ,-0.59316433])
            # observation, reward, done, info = env.step(action)
            if coop:
                done = done['__all__']
        print(f"Trial {i}: Reward = {reward:.2f}")
        all_reward.append(reward)

        # current_dir = os.getcwd()
        # pstate_loc = os.path.join(current_dir, 'Test/Uncovered_States')
        # pathlib.Path(pstate_loc).mkdir(parents=True, exist_ok=True)
        # filename = 'Test'#f'{time.strftime("%Y%m%d-%H%M%S")}'
        # env.set_pstate_file(os.path.join(pstate_loc, filename +".bullet"))

        # if env.save_pstate:
        #     p.saveBullet(env.pstate_file)
        #     env.save_pstate = False
        #     print("SAVED")

        # file = open("saveFile.txt", "w")
        # dumpStateToFile(file)
        # file.close()    

        # #p.restoreState(fileName="20230619-200550.bullet")
        # print("RESTORING")
        # p.restoreState(fileName=os.path.join(pstate_loc, filename +".bullet"))
        # print("RESTORED")

        # stateId = p.saveState()
        # p.restoreState(stateId)


    print("Mean Reward:", np.mean(all_reward))
    print("Reward Std:", np.std(all_reward))
    print(actions)

def dumpStateToFile(file):
    for i in range(p.getNumBodies()):
      pos, orn = p.getBasePositionAndOrientation(i)
      linVel, angVel = p.getBaseVelocity(i)
      txtPos = "pos=" + str(pos) + "\n"
      txtOrn = "orn=" + str(orn) + "\n"
      txtLinVel = "linVel" + str(linVel) + "\n"
      txtAngVel = "angVel" + str(angVel) + "\n"
      file.write(txtPos)
      file.write(txtOrn)
      file.write(txtLinVel)
      file.write(txtAngVel)

def compareFiles(self, file1, file2):
    diff = difflib.unified_diff(
        file1.readlines(),
        file2.readlines(),
        fromfile='saveFile.txt',
        tofile='restoreFile.txt',
    )
    numDifferences = 0
    for line in diff:
      numDifferences = numDifferences + 1
      sys.stdout.write(line)
    self.assertEqual(numDifferences, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)
