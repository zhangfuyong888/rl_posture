from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

import diy_panda_robot 
import diy_panda_task 

import time
import numpy as np

from typing import Optional

class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "nee", # 控制关节角
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        manipulability_value: float = 0.0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = diy_panda_robot.Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = diy_panda_task.Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )



# env = MyRobotTaskEnv(render_mode="human")
# observation, info = env.reset()


# alpha = np.random.rand(7, 1) # 零空间姿态变换向量
# print("零空间向量:", alpha)

# for _ in range(1000):

#     current_position = observation["observation"][0:3]
#     desired_position = observation["desired_goal"][0:3]

#     erro_ee = (0.5 * (desired_position - current_position)).reshape(3, 1) # 末端位置误差做速度向量
#     # print("erro ee:", erro_ee)

#     q_current = np.array([env.robot.get_joint_angle(i) for i in range(7)]) # 当前关节角
#     q = env.robot.inverse_kinematics(link=11, position=desired_position, orientation=np.array([1, 0, 0, 0]))


#     current_J  = env.robot.get_jacobian()[0:3] # 只考虑位置，不考虑姿态
#     # print("雅可比",current_J)

#     J_ = np.linalg.pinv(current_J) # 雅可比伪逆计算
#     # print("雅可比伪逆",J_)

#     J_J = J_ @ current_J
#     # print("雅可比伪逆 * 雅可比",J_J)

#     current_ee_velocity = np.array(env.robot.get_ee_state()[0:3]).reshape(3, 1)
#     # print("末端速度:", current_ee_velocity) 

#     dq = (J_ @ erro_ee) + ((np.eye(7) - J_J) @ alpha)
#     print("关节速度:", dq)

#     observation, reward, terminated, truncated, info = env.step(dq.flatten().tolist())

#     manipulability = env.robot.compute_manipulability()
#     print("操作度",manipulability) # 获取操作度

#     print("reward:", env.task.compute_reward(current_position, desired_position)) # 离散奖励值 -1 


#     if terminated or truncated:
#         observation, info = env.reset()

#     time.sleep(0.1)


# env.close()   










'''验证获取雅可比矩阵的正确性'''
# current_J  = env.robot.get_jacobian()
# print("current_J",current_J)

# current_ee_velocity = env.robot.get_ee_state()
# print("current_ee_velocity:", current_ee_velocity) 

# current_joint_velocity = env.robot.get_joint_velocity()
# print("current_joint_velocity",current_joint_velocity)

# comunicate_ee_velocity = current_J @ current_joint_velocity

# erro_J = current_ee_velocity - comunicate_ee_velocity
# print("erro_J",erro_J)