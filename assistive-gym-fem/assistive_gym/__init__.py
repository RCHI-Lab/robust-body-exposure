from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='RobeReversible-v1',
    entry_point='assistive_gym.envs:RobeReversibleEnv',
    max_episode_steps=200,
)

register(
    id='ReachingPR2-v0',
    entry_point='assistive_gym.envs:ReachingPR2Env',
    max_episode_steps=200,
)
register(
    id='ReachingJaco-v0',
    entry_point='assistive_gym.envs:ReachingJacoEnv',
    max_episode_steps=200,
)
register(
    id='ReachingPR2Human-v0',
    entry_point='assistive_gym.envs:ReachingPR2HumanEnv',
    max_episode_steps=200,
)
register(
    id='ReachingJacoHuman-v0',
    entry_point='assistive_gym.envs:ReachingJacoHumanEnv',
    max_episode_steps=200,
)

register(
    id='DressingPR2Mesh-v1',
    entry_point='assistive_gym.envs:DressingPR2MeshEnv',
    max_episode_steps=200,
)
register(
    id='DressingPR2IKMesh-v1',
    entry_point='assistive_gym.envs:DressingPR2IKMeshEnv',
    max_episode_steps=200,
)
register(
    id='DressingPR2IK-v1',
    entry_point='assistive_gym.envs:DressingPR2IKEnv',
    max_episode_steps=200,
)

register(
    id='BedPosePR2-v1',
    entry_point='assistive_gym.envs:BedPosePR2Env',
    max_episode_steps=200,
)
register(
    id='BedPosePR2Mesh-v1',
    entry_point='assistive_gym.envs:BedPosePR2MeshEnv',
    max_episode_steps=200,
)
register(
    id='BedPoseStretch-v1',
    entry_point='assistive_gym.envs:BedPoseStretchEnv',
    max_episode_steps=200,
)
register(
    id='BedPoseStretchMesh-v1',
    entry_point='assistive_gym.envs:BedPoseStretchMeshEnv',
    max_episode_steps=200,
)


register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

register(
    id='ViewClothVertices-v1',
    entry_point='assistive_gym.envs:ViewClothVerticesEnv',
    max_episode_steps=1000000,
)
#! ADDED FOR GNN TESTING
register(
    id='RobustBodyExposure-v1',
    entry_point='assistive_gym.envs:RobustBodyExposureEnv',
    max_episode_steps=200, #! set to 200 to keep rerolling if grasp not on sheet
)


#! ADDED FOR BEDDDING MANIPULATION
register(
    id='BeddingManipulationSphere-v1',
    entry_point='assistive_gym.envs:BeddingManipulationSphereEnv',
    max_episode_steps=1,
)


register(
    id='BeddingManipulationStretch-v1',
    entry_point='assistive_gym.envs:BeddingManipulationStretchEnv',
    max_episode_steps=200,
)

register(
    id='RemoveContactSphere-v1',
    entry_point='assistive_gym.envs:RemoveContactSphereEnv',
    max_episode_steps=1,
)


register(
    id='HumanSMPLXTesting-v1',
    entry_point='assistive_gym.envs:HumanSMPLXTestingEnv',
    max_episode_steps=200,
)

