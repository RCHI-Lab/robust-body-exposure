import os
from gym import spaces
import numpy as np
import pybullet as p
from .env import AssistiveEnv

class ReachingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False):
        super(ReachingEnv, self).__init__(robot_type=robot_type, task='reaching', human_control=human_control, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=(4 if human_control else 0), obs_robot_len=21, obs_human_len=(19 if human_control else 0))

    def step(self, action):
        # Execute action
        self.take_step(action, robot_arm='right', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.05)

        # Get total force the robot's body is applying to the person
        robot_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            robot_force_on_human += c[9]

        # Get the Euclidean distance between the robot's end effector and the person's mouth
        gripper_pos = np.array(p.getLinkState(self.robot, 54 if self.robot_type=='pr2' else 8, computeForwardKinematics=True, physicsClientId=self.id)[0])
        reward_distance_mouth = -np.linalg.norm(gripper_pos - self.target_pos)
        # Get end effector velocity
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 8, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=robot_force_on_human)
        # Get observations and reward
        obs = self._get_obs([], [robot_force_on_human])
        reward = self.config('distance_weight')*reward_distance_mouth + preferences_score
        info = {'total_force_on_human': robot_force_on_human, 'task_success': int(reward_distance_mouth <= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False
        return obs, reward, done, info

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        robot_right_joint_positions = np.array([x[0] for x in p.getJointStates(self.robot, jointIndices=self.robot_right_arm_joint_indices, physicsClientId=self.id)])
        gripper_pos, gripper_orient = p.getLinkState(self.robot, 54 if self.robot_type=='pr2' else 8, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        if self.human_control:
            human_pos = np.array(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id)[0])
            human_joint_positions = np.array([x[0] for x in p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)])
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        robot_obs = np.concatenate([gripper_pos-torso_pos, gripper_pos-self.target_pos, robot_right_joint_positions, gripper_orient, head_orient, forces]).ravel()
        human_obs = np.concatenate([gripper_pos-human_pos, gripper_pos-self.target_pos, human_joint_positions, gripper_orient, head_orient, forces_human]).ravel() if self.human_control else []
        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        # Create the human, wheelchair, and robot
        self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_right_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_right_arm_joint_indices]
        self.reset_robot_joints()
        if self.robot_type == 'jaco':
            # Attach the Jaco robot to the wheelchair
            wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        # Configure the person
        joints_positions = [(6, np.deg2rad(-90)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
        # Randomize head orientation
        joints_positions += [(21, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (22, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (23, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
        self.human_controllable_joint_indices = [20, 21, 22, 23]
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1], physicsClientId=self.id)
        self.target_human_joint_positions = np.array([x[0] for x in p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        # Create a target on the person's mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        # Change default camera position/angle
        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the starting location of the robot's end effector
        gripper_target_pos = np.array([-0.15, -0.4, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        if self.robot_type == 'pr2':
            # Call self.position_robot_toc() to optimize the robot's base position and then position the end effector using IK.
            gripper_target_orient = p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id)
            self.position_robot_toc(self.robot, 54, [(gripper_target_pos, gripper_target_orient), (self.target_pos, None)], [(self.target_pos, None)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(15, 15+7), pos_offset=np.array([0.1, 0.2, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.03, left=False, set_instantly=True)
        elif self.robot_type == 'jaco':
            # The Jaco is already attached to the wheelchair, so we just need to position the end effector using IK.
            gripper_target_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
            self.util.ik_random_restarts(self.robot, 8, gripper_target_pos, gripper_target_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True, check_env_collisions=True)
            self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)

        # Disable gravity for the robot and human
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs([], [0])

    def update_targets(self):
        # update_targets() is automatically called at each time step for updating any targets in the environment.
        # Move the target marker onto the person's mouth
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)
