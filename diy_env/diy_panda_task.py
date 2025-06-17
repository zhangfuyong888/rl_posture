from typing import Any, Dict, Tuple
import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.5,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.poseture = np.array([0, 0, 0, 0, 0, 0, 0])
        self.target_quat = np.array([0, 0, 0, 1])
        self.line_id =  None

        self.joint_down_limit = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
        self.joint_up_limit = np.array([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671])

        self.target_position = np.array([0.0, 0.0, 0.0])     # 末端位置 (x, y, z)
        self.target_orientation = np.array([0.0, 0.0, 0.0, 0.0])    # 姿态 (四元数)

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0, 
            ghost=True,
            position=np.zeros(3),
            # rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            rgba_color=np.array([0, 191/255.0, 1, 0.5]), # 目标小球位置颜色
        )

    def create_line(self, lineFromXYZ, lineToXYZ):
        # 绘制一条线段
        self.line_id = self.sim.physics_client.addUserDebugLine(
            lineFromXYZ=lineFromXYZ,      # 起点坐标
            lineToXYZ=lineToXYZ,        # 终点坐标
            lineColorRGB=[0, 0, 1],     # 红色 (R,G,B)
            lineWidth=4.0,              # 线宽
            lifeTime=0,                 # 0 表示永久显示，否则为秒数
            parentObjectUniqueId=-1     # 可选：附加到其他对象
        )
        return self.line_id
    
    def create_robot_line(self, lineFromXYZ, lineToXYZ):
        # 绘制一条线段
        robot_line_id = self.sim.physics_client.addUserDebugLine(
            lineFromXYZ=lineFromXYZ,      # 起点坐标
            lineToXYZ=lineToXYZ,        # 终点坐标
            lineColorRGB=[0, 1, 0],     # 红色 (R,G,B)
            lineWidth=4.0,              # 线宽
            lifeTime=0,                 # 0 表示永久显示，否则为秒数
            parentObjectUniqueId=-1     # 可选：附加到其他对象
        )
        return robot_line_id

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal() # 数据格式 [0.20910924 0.08645734 0.27715167]
        # print("goal:", self.goal)

        # rand_euler = np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi])
        # self.target_quat = np.array(self.sim.physics_client.getQuaternionFromEuler(rand_euler)) # 数据格式 (-0.6042888577317447, -0.41570263643832844, -0.5035983479930856, 0.4565249153968723)

        vector_view = self.sim.physics_client.rotateVector(self.target_orientation, np.array([0, 0, 0.2]))    # 将单位向量按照四元素旋转

        if self.line_id  is not  None:
            self.sim.physics_client.removeUserDebugItem(self.line_id)
        self.line_id = self.create_line( self.goal, self.goal + vector_view[0:3])
        # print("vector_view",vector_view)

        # self.poseture = np.append(self.goal,  self.target_orientation)
        # print("poseture",self.poseture) 

        self.sim.set_base_pose("target", self.goal, self.target_orientation)


    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""

        while True:
            goal = self.np_random.uniform(self.joint_down_limit, self.joint_up_limit) 
            # 格式[ 0.90086874 -1.19018702  1.66332655 -1.33214396  1.5554421   1.78292971 1.49760349]

            # Panda 的关节索引（只取前7个用于控制）
            joint_indices = [0, 1, 2, 3, 4, 5, 6]

            # 设置每个关节的角度（不会执行动态，只是设置状态）
            for i, joint_index in enumerate(joint_indices):
                self.sim.physics_client.resetJointState(self.sim._bodies_idx["panda"], joint_index, goal[i])

            # 触发模拟
            self.sim.physics_client.stepSimulation()

            # 检测碰撞
            contacts = self.sim.physics_client.getContactPoints(bodyA=self.sim._bodies_idx["panda"], bodyB=self.sim._bodies_idx["panda"])

            ignored_links = {8, 9, 10}  # 根据你的 URDF，手指的 link index     
            real_contacts = []

            for c in contacts:
                linkA = c[3]
                linkB = c[4]
                if linkA not in ignored_links and linkB not in ignored_links:
                    real_contacts.append(c)

            if len(real_contacts) > 0:
                pass
                # print(" 机器人本体发生自碰撞！")
                # print("碰撞关节：", linkA , linkB)
            else:
                # print(" 无本体碰撞")
                break
        
        # 获取末端执行器的索引
        end_effector_index = 11  # 对于 Panda，末端 link 是 link11

        # 执行正向运动学
        link_state = self.sim.physics_client.getLinkState(self.sim._bodies_idx["panda"], end_effector_index, computeForwardKinematics=True)
        self.target_position = np.array(link_state[0])     # 末端位置 (x, y, z)

        if self.target_position[2] <= 0.3 :
            self.target_position[2] = 0.3

        self.target_orientation = np.array(link_state[1])    # 姿态 (四元数)

        # 输出期望末端位置和四元素
        # print("末端位置：", self.target_position)
        # print("末端四元素：", self.target_orientation)


        # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return self.target_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        flag = np.array(d < self.distance_threshold, dtype=bool)
        # print("flag", flag)
        return flag
    
    def compute_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray :
        d = distance(achieved_goal, desired_goal)
        return d

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray :
        d = distance(achieved_goal, desired_goal)

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)   
        else:
            return -d.astype(np.float32)
