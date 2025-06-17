import gymnasium as gym
import numpy as np
from gymnasium import spaces

import diy_panda_robot 
import diy_panda_task 

from diy_panda_env import MyRobotTaskEnv

from stable_baselines3.common.env_checker import check_env

import time
import os

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.ddpg.policies import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor


class CustomEnv(gym.Env):
    """将pybullet环境转换为gym环境，与stable-line3使用"""
    def __init__(self, render_mode):
        super().__init__()

        # 在之前的类中，初始化pybullet环境 
        self.bs3_env = MyRobotTaskEnv(render_mode=render_mode)
        observation, info = self.bs3_env.reset()
        # 定义动作空间
        self.action_space = self.bs3_env.robot.action_space
        # print("动作空间定义：",self.action_space)
        # 定义观察空间
        self.observation_space = self.bs3_env.observation_space

        self.sim = self.bs3_env.sim
        self.robot = self.bs3_env.robot
        self.task = self.bs3_env.task

        self.max_episode_steps = 500  # 设置最大步数
        self.current_step = 0          # 初始化步数计数器
        self.k_alpha = 1.2             # 设置自由向量alpha系数
        self.robot_line_id = None      # 机器人轨迹线id
        self.reward_total = 0

    def step(self, action: np.ndarray):
        # print(" 输入动作：", action)

        observation = self.bs3_env._get_obs()
        current_position = observation["achieved_goal"][0:3]
        desired_position = observation["desired_goal"][0:3]
        # print("current_position:", current_position, "desired_position:", desired_position)


        if self.task.compute_distance(current_position, desired_position) > 0.5 :
            erro_ee_position = 0.5 * np.array(desired_position - current_position)
            target_orientation = 0.5 * np.array(self.robot.get_ee_orientation_erro(self.bs3_env.task.target_orientation))
        else :
            erro_ee_position = 0.03 * np.array(desired_position - current_position)
            target_orientation = 0.05 * np.array(self.robot.get_ee_orientation_erro(self.bs3_env.task.target_orientation))

        # target_orientation = 0.5 * np.array(self.robot.get_ee_orientation_erro(self.bs3_env.task.target_orientation))
        # erro_ee_position = np.array([0.0, 0.0, 0.0])
        
        erro_ee = np.append(erro_ee_position, target_orientation).reshape(6, 1)  # 测试姿态误差是否正确
        # print("ee_erro:", erro_ee)

        current_J  = self.robot.get_jacobian()
        # print("雅可比",current_J.shape)

        J_ = np.linalg.pinv(current_J) # 雅可比伪逆计算
        # print("雅可比伪逆",J_.shape)

        J_J = J_ @ current_J
        # print("雅可比伪逆 * 雅可比",J_J.shape)

        null_dq = (np.eye(7) - J_J) @ (self.k_alpha * action.reshape(7, 1))
        # print("null_dq", null_dq.shape)

        # dq = (J_ @ erro_ee)     # 不加入零空间雅可比的解

        # dq = (J_ @ erro_ee) + null_dq  # 加入零空间雅可比的解
        # print("关节速度:", dq.shape)

        if self.task.compute_distance(current_position, desired_position) < 0.5 :
            dq = (J_ @ erro_ee)     # 不加入零空间雅可比的解
        else :
            dq = (J_ @ erro_ee) + null_dq  # 加入零空间雅可比的解

        self.robot.set_action(dq.flatten().tolist())
        manipulability = self.robot.compute_manipulability() # 获取当前操作度


        # 机器人姿态可视化
        current_Quaternion = self.robot.ee_Quaternion
        robot_vector_view = self.sim.physics_client.rotateVector(current_Quaternion, np.array([0, 0, 0.2]))
        
        if self.robot_line_id  is not  None:
            self.sim.physics_client.removeUserDebugItem(self.robot_line_id)

        start = self.robot.ee_position[:3]
        end = [s + d for s, d in zip(start, robot_vector_view[:3])]
        self.robot_line_id = self.task.create_robot_line( start, end)

        self.sim.step()
        

        # 保证步数不超过最大值
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            terminated = False
            truncated = True
            info = {"is_success": False}
            reward = -500.0
            self.reward_total += reward
            print("未到目标点,达到最大轮次:", self.current_step)
            return observation, reward, terminated, truncated, info
        
        observation = self.bs3_env._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], observation["desired_goal"][0:3]))
        # print("当前terminated:", terminated)
        truncated = False
        info = {"is_success": terminated}
        if terminated == True:
            # reward = float(self.task.compute_reward(observation["achieved_goal"], observation["desired_goal"][0:3], info))
            reward = 1000 * manipulability
            self.reward_total += reward
            print("已到达目标点,当前轮次:", self.current_step,"manipulability: ", manipulability, "total_reward: ",self.reward_total)
        else :
            reward = float(self.task.compute_reward(observation["achieved_goal"], observation["desired_goal"][0:3], info))
            self.reward_total += reward
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.task.np_random = self.bs3_env.np_random
        self.current_step = 0
        self.reward_total = 0
        # print("轮次重置")
        with self.sim.no_rendering():
            self.sim.physics_client.removeAllUserDebugItems()
            self.task.reset()
            self.robot.reset() 
        observation = self.bs3_env._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def render(self):
        return self.sim.render(
            width=self.bs3_env.render_width,
            height=self.bs3_env.render_height,
            target_position=self.bs3_env.render_target_position,
            distance=self.bs3_env.render_distance,
            yaw=self.bs3_env.render_yaw,
            pitch=self.bs3_env.render_pitch,
            roll=self.bs3_env.render_roll,
        )

    def close(self):
        self.sim.close()


if __name__ == "__main__":

    state = "test" # train or test

    if state == "train":
        # 统一输出根目录
        current_path = os.path.dirname(os.path.abspath(__file__))
        output_root = os.path.join(current_path, "output")
        print( "output_root:", output_root)
        tensorboard_dir = os.path.join(output_root, "tensorboard")
        print("tensorboard_dir:", tensorboard_dir)
        models_dir = os.path.join(output_root, "models")
        print("models_dir", models_dir)
        logs_dir = os.path.join(output_root, "logs")
        best_model_dir = os.path.join(models_dir, "best_model")


        env = CustomEnv(render_mode= "rgb_array") # 可视化 "human" or "rgb_array"
        check_env(env)

        # 添加Monitor包装器
        env = Monitor(env)

        observation, info = env.reset()
        # 定义动作噪声（DDPG需要探索噪声）
        n_actions = env.action_space.shape[0]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        # 或者使用OU噪声：
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


        # 创建DDPG模型   第一次训练需要创建DDPG模型
        model = DDPG(
            "MultiInputPolicy",                      # 多层感知机策略网络
            env,
            action_noise=None,                       # 添加动作噪声   当前无噪声
            verbose=1,                               # 打印训练信息
            tensorboard_log=tensorboard_dir,         # 保存TensorBoard日志
            learning_starts=5000,       # 随机动作步数，用于初始化经验回放
            buffer_size=100000,         # 经验回放缓冲区大小
            batch_size=2048,             # 训练批次大小
            gamma=0.99,                 # 折扣因子
            tau=0.005,                  # 目标网络软更新参数
            learning_rate=1e-3,         # 学习率
            train_freq=(10, "step"),
            gradient_steps=10,
            device="cuda",
        )
        

        # # 加载模型  用新模型训练时，注释掉上面创建的DDPG模型就行
        # model = DDPG.load(best_model_dir, env=env)
        # # replay_buffer_path = os.path.join(models_dir, "poseture_opt_panda_replay_buffer_10000_steps") # 加载经验池 后缀为 .pkl
        # # model.load_replay_buffer(replay_buffer_path)


        # 设置回调函数
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=models_dir,
            name_prefix="poseture_opt",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # 评估回调(在验证环境上定期评估)
        eval_callback = EvalCallback(
            env,
            best_model_save_path=best_model_dir,
            log_path=logs_dir,
            eval_freq=1000,                         # 记录频率
            deterministic=True,
            render=False,
        )
        

        start_time = time.time()  # 记录开始时间
        # 训练模型
        model.learn(
            total_timesteps=500000,  # 总训练步数
            callback=[checkpoint_callback, eval_callback ],  # 应用回调
            tb_log_name="panda_reach",
            log_interval=100,  # 日志打印间隔,每隔10 episodes
            progress_bar=True,
        )

        # 保存最终模型
        model.save(os.path.join(output_root, "ddpg_panda_final"))

        # 评估模型
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        # 打印训练总耗时
        print(f"训练耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")

        env.close()


    elif state == "test":
        # 创建渲染模式为"human"的环境（显示PyBullet窗口）
        env = CustomEnv(render_mode= "human")

        
        # 加载训练好的模型
        model = DDPG.load("/home/zfy/opt_poseture/diy_env/output/ddpg_panda_final.zip", env=env)  # 替换为你的模型路径
        
        print("开始可视化测试...")
        obs, info = env.reset()
        
        for episode in range(10):  # 测试5个episode
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            test_step = 0
            while not (terminated or truncated):
                # 获取模型动作（deterministic=True关闭探索噪声）
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                env.render()  # 渲染每一帧
                # time.sleep(0.01)
                test_step += 1
                if  test_step  >= 300 :  # 最大步数 跳出循环
                    print("达到最大步数")
                    break
                
            print(f"Episode {episode+1} 奖励: {episode_reward:.2f}, 目标点: {info['is_success']}")
        
        env.close()






