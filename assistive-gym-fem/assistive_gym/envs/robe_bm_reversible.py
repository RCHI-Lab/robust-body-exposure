import os
import pickle
import time

import cv2
import numpy as np
import pybullet as p

# since there is no gnn_testing_envs file
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from .bu_gnn_util import *
from .env import AssistiveEnv
import pickle, pathlib
from pathlib import Path
import os.path as osp

human_controllable_joint_indices = []

#! max_episode_steps=200 CHECK THAT THIS IS STILL OKAY MOVING FORWARD

class RobeReversibleEnv(AssistiveEnv):
    def __init__(self):
        obs_robot_len = 28
        self.single_ppo_model = False # will randomize target limb

        self.min_threshold = .05
        self.max_threshold = .06
        self.line_threshold = .1
        self.fixed_body_height = True
        
        self.data_intermediate = []

        self.release_threshold = .2
        self.lower_before_release = False

        self.points_pos_limb_world = []
        self.render_body_points = True

        self.ppo = False
        if self.single_ppo_model:
            obs_robot_len = 28 + len(all_possible_target_limbs)

        self.iteration = 0
        
        super(RobeReversibleEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='bedding_manipulation', obs_robot_len=obs_robot_len, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)
        self.use_mesh = False
        self.reverse_action = False

        self.save_pstate = True
        self.pstate_file = None

        self.take_pictures = True
        self.rendering = False
        self.target_limb_code = 5 if not self.single_ppo_model else None # randomize in reset function, None acts as a placeholder
        self.fixed_pose = False

        if self.ppo:
            self.collect_data = False
            self.blanket_pose_var = False
            self.high_pose_var = False
            self.body_shape_var = False
            self.body_shape = None if self.body_shape_var == True else np.zeros((1, 10))
            self.gender = 'random' if self.body_shape_var == True else 'female'
        else:
            self.collect_data = None
            self.blanket_pose_var = None
            self.high_pose_var = None
            self.body_shape_var = None

        self.naive = False
        self.clip = True #! Turn off for cma-data collect

        self.human_no_occlusion_RGB = None
        self.human_no_occlusion_depth = None
        self.point_cloud_initial = None
        self.point_cloud_final = None
        self.point_cloud_depth_img = None

        self.new_anchors = False
        self.seed_val = 100
        self.cma_sim_dyn = False

        self.take_images = True
    
    def set_seed_val(self, seed = 1001):
        # if seed != self.seed_val:
        #     self.seed_val = seed
        self.seed_val = seed

    def set_target_limb_code(self, code):
        self.target_limb_code = code
    
    def set_env_variations(self, collect_data, blanket_pose_var, high_pose_var, body_shape_var):
        self.collect_data = collect_data
        self.blanket_pose_var = blanket_pose_var
        self.high_pose_var = high_pose_var
        self.body_shape_var = body_shape_var

        self.body_shape = None if self.body_shape_var == True else np.zeros((1, 10))
        self.gender = 'random' if self.body_shape_var == True else 'female'

    def get_human_body_info(self):
        return self.human_creation.body_info if self.body_shape_var else None
        
    def step(self, action):
        time_var = time.strftime("%Y%m%d-%H%M%S")

        self.execute_action = True
        obs = self._get_obs()

        # TODO: only used when training PPO policies for the real world, change execution conditions
        if self.ppo:
            action = remap_action_ppo(action, remap_ranges=[[0, 1], [-0.5, 1], [0, 1], [-1, 1]])
        
        # if self.rendering:
        #     print(obs)
        #     print(action)
        #     print(self.target_limb_code)

        # * scale bounds the 2D grasp and release locations to the area over the mattress (action nums only in range [-1, 1])
        # * if using the naive approach, do not scale the action since it is determined directly from points over the bed
        action = scale_action(action) if not self.naive else scale_action(action, scale=[1, 1])

        grasp_loc = action[0:2]
        release_loc = action[2:4]

        # * get points on the blanket, initial state of the cloth
        data_initial = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        # TODO: increase clipping threshold to fit the new cloth
        dist, is_on_cloth = check_grasp_on_cloth(action, np.array(data_initial[1]), clipping_thres=.028)
    
        # * if no points on the blanket are within 2.8 cm of the grasp location, exit (if collecting data) or proceed without executing the action (in all other conditions)
        if not is_on_cloth:
            if self.collect_data:
                return obs, 0, False, {} # for data collect
            elif self.clip:
                self.execute_action = False

        if self.execute_action:
            if self.take_images: self.capture_images("initial", self.iteration, self.min_threshold*100)

            # Get the 4 closest distance indexes to serve as anchor points
            anchor_idx = np.argpartition(np.array(dist), 4)[:4]

            # * update grasp_loc var with the location of the central anchor point on the cloth
            # Get the vertices on the blanket mesh and pull the x,y corresponding to the first anchor index
            grasp_loc = np.array(data_initial[1][anchor_idx[0]][0:2])

            # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
            self.sphere_ee.set_base_pos_orient(data_initial[1][anchor_idx[0]], np.array([0,0,0]))
            constraint_ids = []
            #Hold the first anchor
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
            for i in anchor_idx[1:]:
                #Get the mesh vertex x,y corresponding to the anchor id index and subtract the 0th index
                #Gives the relative position to the first anchor index
                pos_diff = np.array(data_initial[1][i]) - np.array(data_initial[1][anchor_idx[0]])
                #Add the other anchors
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))
                        
            # * move sphere up by some delta z
            # TODO: look at making this a fixed delta z from the body, not the bed
            #           will need to adjust the z position at each step
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            if self.fixed_body_height:
                delta_z = self.min_threshold             
            else:
                delta_z = .4           # distance to move up (with respect to the top of the bed)
            bed_height = 0.58                       # height of the bed
            final_z = delta_z + bed_height          # global goal z position

            #Moves sphere up to the height
            while current_pos[2] <= final_z:
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            # * move sphere to the release location, release the blanket
            travel_dist = release_loc - grasp_loc

            # * determine delta x and y, make sure it is, at max, close to 0.005
            num_steps = np.abs(travel_dist//0.005).max()
            delta_x, delta_y = travel_dist/num_steps

            # * if no points on the blanket are within 2.8 cm of the grasp location, clip
            #is_on_cloth = (np.any(np.array(dist) < clipping_thres)) 
    
            delta_z = 0
            current_pos = self.sphere_ee.get_base_pos_orient()[0]

            #Moves the sphere to the release location           
            for _ in range(int(num_steps)):
                if self.fixed_body_height:
                    delta_z = check_height_of_effector(np.array(current_pos), np.array(self.points_pos_limb_world), self.min_threshold, self.max_threshold, self.line_threshold)      
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, delta_z]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                
            # * continue stepping simulation to allow the cloth to settle before release
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.id)

            if self.lower_before_release:
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                delta_z = release_height_of_effector(np.array(current_pos), np.array(self.points_pos_limb_world), self.release_threshold)      
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, delta_z]), np.array([0,0,0]))
                for _ in range(50):
                    p.stepSimulation(physicsClientId=self.id)

            # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
            for i in constraint_ids:
                p.removeConstraint(i, physicsClientId=self.id)
            for _ in range(50):
                p.stepSimulation(physicsClientId=self.id)

            # * get points on the blanket, intermediate state of the cloth
            self.data_intermediate = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

##########Perform second action
            if self.reverse_action:
                #Reverse the actions 
                grasp_loc = action[2:4]
                release_loc = action[0:2]
                action =  [*grasp_loc, *release_loc]

                if self.new_anchors:
                    # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
                    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(self.data_intermediate[1]))
                    # * if no points on the blanket are within 2.8 cm of the grasp location, exit (if collecting data) or proceed without executing the action (in all other conditions)
                    if not is_on_cloth:
                        if self.collect_data:
                            return obs, 0, False, {} # for data collect
                        elif self.clip:
                            self.execute_action = False

                    if self.execute_action:
                        anchor_idx = np.argpartition(np.array(dist), 4)[:4] # Finding set of points closest to grasp point

                # * update grasp_loc var with the location of the central anchor point on the cloth
                grasp_loc = np.array(self.data_intermediate[1][anchor_idx[0]][0:2])

                # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
                self.sphere_ee.set_base_pos_orient(self.data_intermediate[1][anchor_idx[0]], np.array([0,0,0]))
                constraint_ids = []
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
                for i in anchor_idx[1:]:
                    pos_diff = np.array(self.data_intermediate[1][i]) - np.array(self.data_intermediate[1][anchor_idx[0]])
                    constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))
                
                # * move sphere up by some delta z
                # TODO: look at making this a fixed delta z from the body, not the bed
                #           will need to adjust the z position at each step
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                delta_z = .4#self.min_threshold                        # distance to move up (with respect to the top of the bed)
                bed_height = 0.58                       # height of the bed
                final_z = delta_z + bed_height          # global goal z position

                #Moves sphere up to the height
                while current_pos[2] <= final_z:
                    self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
                    p.stepSimulation(physicsClientId=self.id)
                    current_pos = self.sphere_ee.get_base_pos_orient()[0]

                # * move sphere to the release location, release the blanket
                travel_dist = release_loc - grasp_loc

                # * determine delta x and y, make sure it is, at max, close to 0.005
                num_steps = np.abs(travel_dist//0.005).max()
                delta_x, delta_y = travel_dist/num_steps

                # * if no points on the blanket are within 2.8 cm of the grasp location, clip
                #is_on_cloth = (np.any(np.array(dist) < clipping_thres)) 
        
                delta_z = 0
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

                #Moves the sphere to the release location           
                for _ in range(int(num_steps)):
                    if self.fixed_body_height:
                        delta_z = check_height_of_effector(np.array(current_pos), np.array(self.points_pos_limb_world), self.min_threshold, self.max_threshold)      
                    self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, delta_z]), np.array([0,0,0]))
                    p.stepSimulation(physicsClientId=self.id)
                    current_pos = self.sphere_ee.get_base_pos_orient()[0]
                
                # * continue stepping simulation to allow the cloth to settle before release
                for _ in range(20):
                    p.stepSimulation(physicsClientId=self.id)

                # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
                for i in constraint_ids:
                    p.removeConstraint(i, physicsClientId=self.id)
                for _ in range(50):
                    p.stepSimulation(physicsClientId=self.id)

        # * get points on the blanket, final state of the cloth
        data_final = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        reward = 0
        cloth_initial_subsample = cloth_final_subsample = -1 # none for data collection
        info = {}
        
        if self.take_images: self.capture_images("final", self.iteration, self.min_threshold*100)

        human_pose = np.reshape(self.human_pose, (-1,2))
        all_body_points = get_body_points_from_obs(human_pose, target_limb_code=self.target_limb_code, body_info=self.get_human_body_info())
            
        if not self.collect_data:
            # TODO - REPLACE REWARD CALCULATION HERE WTIH FUNCTION FROM GNN_UTIL
            #cloth_initial_subsample, cloth_intermediate_subsample, cloth_final_subsample = sub_sample_point_clouds(data_initial[1], data_initial[1], data_final[1])
            cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(data_initial[1], data_final[1])

            cloth_initial_2D = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
            #cloth_intermediate_2D = np.delete(np.array(cloth_intermediate_subsample), 2, axis = 1)
            cloth_final_2D = np.delete(np.array(cloth_final_subsample), 2, axis = 1)

            human_pose = np.reshape(self.human_pose, (-1,2))

            all_body_points = get_body_points_from_obs(human_pose, target_limb_code=self.target_limb_code, body_info=self.get_human_body_info())
            reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
            info = {
                "cloth_initial": data_initial,
                #"cloth_intermediate" : self.data_intermediate,
                "cloth_final": data_final,
                "RBG_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "cloth_initial_subsample": cloth_initial_subsample,
                #"cloth_intermediate_subsample" : cloth_intermediate_subsample,
                "cloth_final_subsample": cloth_final_subsample,
                "covered_status_sim": covered_status,
                "target_limb_code":self.target_limb_code,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender,
                "grasp_on_cloth":self.execute_action
                }
        else:
            info = {
                "cloth_initial": data_initial,
                #"cloth_intermediate" : self.data_intermediate,
                "cloth_final": data_final,
                "RGB_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "point_cloud_depth_img": self.point_cloud_depth_img,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender,
                "all_body_points": all_body_points
                }
        # print("INFO", info)
        # current_dir = os.getcwd()
        # pstate_loc = os.path.join(current_dir, 'Test/Uncovered_States')
        # pathlib.Path(pstate_loc).mkdir(parents=True, exist_ok=True)
        # filename = f'{time.strftime("%Y%m%d-%H%M%S")}'
        # self.set_pstate_file(os.path.join(pstate_loc, filename +".bullet"))

        # if self.save_pstate:
        #     p.saveBullet(self.pstate_file)
        #     self.save_pstate = False

        self.iteration += 1
        done = self.iteration >= 1

        # return 0, 0, 1, {}
        return obs, reward, done, info
    
    def set_pstate_file(self, filename):
        if self.pstate_file != filename:
            self.pstate_file = filename
            self.save_pstate = True

    def get_cloth_state(self):
        return p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]

    def _get_obs(self, agent=None):

        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]        
        pose = []
        for limb in self.human.obs_limbs:
            pos, orient = self.human.get_pos_orient(limb)
            pos2D = pos[0:2]
            # yaw = p.getEulerFromQuaternion(orient)[-1]
            # pose.append(np.concatenate((pos2D, np.array([yaw])), axis=0))
            pose.append(pos2D)
        pose = np.concatenate(pose, axis=0)
        self.human_pose = pose

        if self.collect_data:

            output = [None]*28
            all_joint_angles = self.human.get_joint_angles(self.human.all_joint_indices)
            all_pos_orient = [self.human.get_pos_orient(limb) for limb in self.human.all_body_parts]
            output[0], output[1], output[2] = pose, all_joint_angles, all_pos_orient
            return output

        if self.single_ppo_model:
            one_hot_target_limb = [0]*len(self.human.all_possible_target_limbs)
            one_hot_target_limb[self.target_limb_code] = 1
            pose = np.concatenate([one_hot_target_limb, pose], axis=0)
        return np.float32(pose)

    def reset(self):
        if self.cma_sim_dyn:
            self.seed(self.seed_val)

        super(RobeReversibleEnv, self).reset()

        #Create new human each time if using mesh
        if self.use_mesh:
            self.human = human=Human(human_controllable_joint_indices, controllable=True)

        self.build_assistive_env(fixed_human_base=False, gender=self.gender, human_impairment='none', furniture_type='hospital_bed', body_shape=self.body_shape)

        self.target_limb_code = self.target_limb_code if not self.single_ppo_model else randomize_target_limbs()

        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        
        # * Setup human in the air, with legs and arms slightly seperated
        joints_positions = [(self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]

        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)

        # move the human to new location above the bed (position, orientation)
        self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])
        random_variation = []
        if not self.fixed_pose:
            # * Add small variation to the body pose
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            random_variation = self.np_random.uniform(-.2, .2, size=len(motor_indices))
            self.human.set_joint_angles(motor_indices, motor_positions + random_variation)

            if self.high_pose_var:
                self.increase_pose_variation()
            # * Increase friction of joints so human doesn't fail around exessively as they settle
            # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
            self.human.set_whole_body_frictions(spinning_friction=2, lateral_friction=.6)
        
        # * Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        # * step the simulation a few times so that the human has some initial velocity greater than the at rest threshold
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)
        # * continue stepping the simulation until the human joint velocities are under the threshold
        threshold = 1e-2
        settling = True
        numsteps = 0
        while settling:
            settling = False
            for i in self.human.all_joint_indices:
                if np.any(np.abs(self.human.get_velocity(i)) >= threshold):
                    p.stepSimulation(physicsClientId=self.id)
                    numsteps += 1
                    settling = True
                    break
            if numsteps > 400:
                break
        # print("steps to rest:", numsteps)

        # * Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        self.generate_points_along_body()

##########Using mesh#################
        if self.use_mesh:
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states(self.human.all_joint_indices)

            # Replace the capsulized human with a human mesh
            self.human_mesh = HumanMesh()
            # TODO: Change joint positions so that you're using the joint positions of the capsulized human
            #joints_positions = [(self.human.j_left_hip_z, 10), (self.human.j_right_hip_z, -10), (self.human.j_left_shoulder_z, -50), (self.human.j_right_shoulder_z, 50)]
            
            joints_positions = self.human_mesh.convert(self.human, random_variation)
            #self.human_mesh.obs_limbs_conversion(self.human)

            #Move capsulized human to the side
            self.human.set_base_pos_orient([-5, -0.5, .7], [-np.pi/2.0, 0, np.pi])
            #Reasign human variable to the mesh
            self.human = self.human_mesh
            #joints_positions = np.array(motor_positions) * 10#+ self.np_random.uniform(-10, 10, size=len(motor_indices))

            # TODO: use body_shape param in the enviornment instead of this fixed array, same for gender
            body_shape = self.body_shape #np.zeros((1, 10))
            gender = self.gender #'female' # 'random'

            # TODO: delete the capsulized person to finish this out (delete before initializing the mesh)
            self.human.init(self.directory, self.id, self.np_random, reverse_env=True, gender=self.gender, height=None, body_shape=body_shape, joint_angles=joints_positions)

            # TODO: adjust spawn location so that human lays in the bed - may not be straightforward
            #chair_seat_position = np.array([0, 0.1, 0.55])
            self.human_mesh.set_base_pos_orient([0, -0.5, .7], [-np.pi/2.0, 0, np.pi])
            #self.human_mesh.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])

            #self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

        # * Setup camera for taking images
        # *     Currently saves color images only to specified directory
        if self.take_pictures or self.collect_data:
            self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=398)
            img, depth = self.get_camera_image_depth()
            self.human_no_occlusion_RGB = img
            self.human_no_occlusion_depth = depth
            depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
            depth = (depth * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
            # filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
            # filename = "test_depth.png"
            # cv2.imwrite(os.path.join('/home/mycroft/git/vBMdev/pose_variation_images/lower_var2', filename), img)
            # cv2.imwrite(filename, depth_colormap)

        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_1061v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, .6], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)

        # * can apply some variation in the blanket's initial position, otherwise configure over the person so that they are covered up to the shoulders/neck
        if self.blanket_pose_var:
            delta_y = self.np_random.uniform(-0.25, 0.05)
            delta_x = self.np_random.uniform(-0.02, 0.02)
            deg = 45
            delta_rad = self.np_random.uniform(-np.radians(deg), np.radians(deg)) # * +/- degrees
            p.resetBasePositionAndOrientation(self.blanket, [0+delta_x, 0.2+delta_y, 1.5], self.get_quaternion([np.pi/2.0, 0, 0+delta_rad]), physicsClientId=self.id)
        else:
            p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, .6], flags=0, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # time.sleep(2)

        # return 0


        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # self.non_target_initially_uncovered(data)
        # self.uncover_nontarget_reward(data)

    
        # * Initialize enviornment variables
        # *     if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
        # self.time = time.time()
        if self.robot is None:
            # * spawn sphere manipulator
            # position = np.array([-0.3, -0.86, 0.8])
            position = np.array([2 ,2, 0]) # move out of the way so it doesn't interfere with the initial depth image
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])
            
            # * initialize env variables
            from gym import spaces
            # * update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)
            # * Define action/obs lengths
            self.action_robot_len = 4
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()
        

        return self._get_obs()
    
    def generate_points_along_body(self):
        '''
        generate all the target/nontarget posistions necessary to uniformly cover the body parts with points
        if rendering, generates sphere bodies as well
        '''

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0

        self.points_pos_on_nontarget_limb = {}
        self.points_nontarget_limb = {}
        self.total_nontarget_point_count = 0

        #! just for tuning points on torso
        # self.human.all_body_parts = [self.human.waist]

        # * create points on all the body parts
        for limb in self.human.all_body_parts:

            # * get the length and radius of the given body part
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]

            # * create points seperately depending on whether or not the body part is/is a part of the target limb
            # *     generates list of point positions around the body part capsule (sphere if the hands)
            # *     creates all the spheres necessary to uniformly cover the body part (spheres created at some arbitrary position (transformed to correct location in update_points_along_body())
            # *     add to running total of target/nontarget points
            # *     only generate sphere bodies if self.rendering == True
            if hasattr(self.human, "target_limb"):
                if limb in self.target_limb:
                    if limb in [self.human.left_hand, self.human.right_hand]:
                        self.points_pos_on_target_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                    else:
                        self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                    if self.rendering:
                        self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                    self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_nontarget_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.rendering:
                    self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 1, 0, 0.2])
                self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])
                # print("COUNT", self.total_nontarget_point_count)

        # * transforms the generated spheres to the correct coordinate space (aligns points to the limbs)
        self.update_points_along_body()
    
    def update_points_along_body(self):
        '''
        transforms the target/nontarget points created in generate_points_along_body() to the correct coordinate space so that they are aligned with their respective body part
        if rendering, transforms the sphere bodies as well
        '''

        # * positions of the points on the target/nontarget limbs in world coordinates
        self.points_pos_target_limb_world = {}
        self.points_pos_nontarget_limb_world = {}

        # * transform all spheres for all the body parts
        for limb in self.human.all_body_parts:

            # * get current position and orientation of the limbs, apply a correction to the pos, orient if necessary
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            
            # * transform target/nontarget point positions to the world coordinate system so they align with the body parts

            if hasattr(self.human, "target_limb"):
                if limb in self.target_limb:
                    for i in range(len(self.points_pos_on_target_limb[limb])):
                        point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_target_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                        self.points_pos_limb_world.append(point_pos)
                        if self.rendering:
                            self.points_target_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                    self.points_pos_target_limb_world[limb] = self.points_pos_limb_world
            else:
                for i in range(len(self.points_pos_on_nontarget_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_nontarget_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    self.points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_nontarget_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = self.points_pos_limb_world


    def increase_pose_variation(self):
        '''
        Allow more variation in the knee and elbow angles
          can be some random position within the lower and upper limits of the joint movement (range is made a little smaller than the limits of the joint to prevent angles that are too extreme)
        '''
        for joint in (self.human.j_left_knee, self.human.j_right_knee, self.human.j_left_elbow, self.human.j_right_elbow):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states([joint])
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(self.human.lower_limits[joint]+0.1, self.human.upper_limits[joint]-0.1, 1))

    def capture_images(self, state, iteration, height):
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=410)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        var_type = f"Test"
        filename = f'top_view_{state}_{iteration}.png'
        img_dir = osp.join(os.getcwd(), var_type)
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, filename), img)

    def set_iteration(self, seed):
        self.iteration = seed


