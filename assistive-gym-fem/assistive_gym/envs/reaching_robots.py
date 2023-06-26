from .reaching import ReachingEnv

class ReachingPR2Env(ReachingEnv):
    def __init__(self):
        super(ReachingPR2Env, self).__init__(robot_type='pr2', human_control=False)
class ReachingJacoEnv(ReachingEnv):
    def __init__(self):
        super(ReachingJacoEnv, self).__init__(robot_type='jaco', human_control=False)
class ReachingPR2HumanEnv(ReachingEnv):
    def __init__(self):
        super(ReachingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)
class ReachingJacoHumanEnv(ReachingEnv):
    def __init__(self):
        super(ReachingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)
