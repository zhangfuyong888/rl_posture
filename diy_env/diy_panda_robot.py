from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

from scipy.spatial.transform import Rotation as R

class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),


        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        self.ee_position = np.array([0.0,0.0,0.0])  # 坐标系中的位置 (x, y, z)
        self.ee_Quaternion = np.array([0.0,0.0,0.0,0.0])  # Quaternion (x, y, z, w)

        self._manipulability_text_id = None

        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))

        self.control_joints(target_angles=target_angles)


    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        arm_joint_ctrl = arm_joint_ctrl   # 控制量就是速度
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)
    

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
    
    def get_joint_velocity(self) -> np.ndarray:
        """Returns the velocity of the joint as (vx, vy, vz)"""

        joint_velocities = [ self.sim.physics_client.getJointState(self.sim._bodies_idx[self.body_name], i)[1] for i in self.joint_indices ]

        return joint_velocities[0:7]
    
    def get_ee_orientation_erro(self, desire_quaternion) -> np.ndarray:
        """返回末端姿态误差"""

        link_state = self.sim.physics_client.getLinkState(
            bodyUniqueId=self.sim._bodies_idx[self.body_name],
            linkIndex=self.ee_link,
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )

        # Orientation (世界坐标系下的姿态四元数)
        self.ee_position = link_state[0]  # 坐标系中的位置 (x, y, z)
        self.ee_Quaternion = link_state[1]  # Quaternion (x, y, z, w)
        current_quaternion = np.array(self.ee_Quaternion)

        # 对偶性处理，避免跳变（四元数与目标对齐）
        if np.dot(current_quaternion, np.array(desire_quaternion)) < 0.0:
            current_quaternion *= -1
        # print("current_quaternion:",current_quaternion,"desire_quaternion:",desire_quaternion)

        # _ , ee_Quaternion_inv = self.sim.physics_client.invertTransform([0, 0, 0], current_quaternion)
        # print("ee_Quaternion_inv:",ee_Quaternion_inv)

        # 构造旋转对象
        q_current = R.from_quat(current_quaternion)  # (x, y, z, w)
        q_desired = R.from_quat(desire_quaternion)
        # print("q_current:",q_current.as_quat(),"q_desired:",q_desired.as_quat())

        # 误差四元数：当前的逆 × 目标
        q_error = q_current.inv() * q_desired
        # print("q_error:",q_error.as_quat())

        # 将误差旋转的前三个元素（向量部分）变换到世界坐标系下
        rot_matrix = np.array(self.sim.physics_client.getMatrixFromQuaternion(self.ee_Quaternion)).reshape(3, 3)
        q_error_vec = 1 * rot_matrix @ q_error.as_quat()[:3].reshape(3, 1)

        # 重新构造一个四元数（向量部分 + 标量部分）
        q_error_3 = np.vstack((q_error_vec, q_error.as_quat()[3]))  # shape (4, 1)

        # 转为 Rotation 对象再转欧拉角
        q_error_rot = R.from_quat(q_error_3.flatten())  # flatten to (4,)
        error_euler = q_error_rot.as_euler('xyz', degrees=False)
        # print("欧拉角误差 :", error_euler)

        return error_euler

    
    def get_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 Panda 末端执行器的线速度与角速度雅可比矩阵"""
        robot_id = self.sim._bodies_idx[self.body_name]
        # print("robot_id",robot_id)

        num_joints = self.sim.physics_client.getNumJoints(robot_id)
        # print(f"机器人总关节数: {num_joints}")

        link_index = self.ee_link
        # print("link_index",link_index)
        local_position = [0, 0, 0]

        #  重要：使用所有控制关节（包括手指），9个关节
        joint_indices = self.joint_indices  # [0,1,2,3,4,5,6,9,10]
        # print("joint_indices",joint_indices)
        n_dof = len(joint_indices)
        # print("n_dof",n_dof)

        # 获取所有控制关节的状态（角度、速度、加速度）
        joint_angles = [self.sim.get_joint_angle(self.body_name, i) for i in joint_indices]
        # print("joint_angles",joint_angles)
        joint_velocities = [0.0] * n_dof
        joint_accelerations = [0.0] * n_dof

        # 调用 PyBullet 的雅可比计算
        J_lin, J_ang = self.sim.physics_client.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=link_index,
            localPosition=local_position,
            objPositions=joint_angles,
            objVelocities=joint_velocities,
            objAccelerations=joint_accelerations
        )

        # 保留前7列（主机械臂关节）
        J_full = np.vstack((J_lin, J_ang))
        J_reduced = J_full[:, :7]  # shape: (6, 7)
        return J_reduced

    def compute_manipulability(self):

        J = self.get_jacobian()
        # print("J",J)

        # J_ = np.linalg.pinv(J) # 雅可比伪逆计算
        # print("J_",J_)

        JJ_T = J @ J.T  # 6 x 6
        w = np.sqrt(np.linalg.det(JJ_T))

        # 如果已有文字显示，先移除旧的
        if self._manipulability_text_id !=  None:
            self.sim.physics_client.removeUserDebugItem(self._manipulability_text_id)

        # 添加新的调试文本
        self._manipulability_text_id = self.sim.physics_client.addUserDebugText(
            text="manipulability:{:.6f}".format(w),
            # textPosition=self.get_ee_position() + np.array([0, 0, 0.1]),  # 显示在末端上方一点
            textPosition=np.array([0.2, 0.2, 0.2]),  # 显示在末端上方一点
            textColorRGB=[0, 1, 0],
            textSize=2,
            lifeTime=0  # 永久显示直到手动移除
        )
        return w

    def get_robot_joint_limit(self, ):
        # 获取机器人关节数量
        num_joints = self.sim.physics_client.getNumJoints(self.sim._bodies_idx[self.body_name])

        print("机器人关节限幅：")
        for joint_index in range(num_joints):
            joint_info = self.sim.physics_client.getJointInfo(self.sim._bodies_idx[self.body_name], joint_index)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            # 仅对可移动关节显示限幅（非固定关节）
            if joint_type != self.sim.physics_client.JOINT_FIXED:
                print(f"关节 {joint_index} ({joint_name}): 下限 = {lower_limit}, 上限 = {upper_limit}")

