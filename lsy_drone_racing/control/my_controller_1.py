from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class MyController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        self.actual_trajectory = []
        self.init_pos = obs['pos']
        self.control_tick = 5

    def _augment_gate_waypoints(self,
                            drone_pos: np.ndarray,
                            gates_pos: NDArray[np.floating], 
                            gates_quat: NDArray[np.floating],
                            obstacles_pos: NDArray[np.floating],
                            target_gate_idx: int) -> NDArray[np.floating]:  # 新增目标门参数
        """根据当前target_gate生成穿门轨迹"""
        
        augmented_waypoints = [self.init_pos]
        curr_ind = 0

        for i in range(len(gates_pos)):
            if i < target_gate_idx:
                augmented_waypoints.extend([gates_pos[i]])

            elif i == target_gate_idx and i != 0:
                augmented_waypoints.extend([drone_pos])
                curr_ind = i

            else:
                current_gate = gates_pos[i]

                # 动态计算门朝向
                if i > target_gate_idx:
                    # 当前目标门：朝向无人机来向
                    approach_vector = drone_pos - current_gate
                else:
                    # 后续门：指向前一个已处理门的后方
                    approach_vector = gates_pos[i-1] - current_gate
                
                # 处理门朝向向量
                gate_forward = R.from_quat(gates_quat[target_gate_idx]).apply([0, 1, 0])
                if np.dot(gate_forward, approach_vector) > 0:
                    gate_forward = -gate_forward
            
                pre_gate = current_gate + gate_forward * 1.0  # 门前1米

                post_gate = current_gate - gate_forward * 2.0  # 门后1米

                augmented_waypoints.extend([pre_gate, post_gate])
        
        return np.array(augmented_waypoints), curr_ind

    def plot_waypoints(self, original_points: np.ndarray, smooth_points: np.ndarray, actual_points, idx: int):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 原始轨迹（红色折线）
        ax.plot(original_points[:, 0], original_points[:, 1], original_points[:, 2], 'ro-', label='Original')

        # 平滑轨迹（蓝色曲线）
        ax.plot(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], 'b-', label='Smoothed')
        for i, (x, y, z) in enumerate(original_points):
            ax.text(x, y, z, f'{i}', color='black', fontsize=8)

        if actual_points is not None and len(actual_points) > 0:
            ax.plot(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
                    'g--', label='Actual Flight', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Visualization')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'fig{idx}.png')

    def _get_gate_forward_vector(self, gate_quat: np.ndarray) -> np.ndarray:
        """获取门的正前方向量"""
        rot = R.from_quat(gate_quat)
        return rot.apply([1, 0, 0])  # 假设门的局部x轴是前方

    def replan(self, waypoints: NDArray[np.floating]):
        """重新规划轨迹，基于输入的 waypoints"""
        self.num_points = waypoints.shape[0]
        self.t_total = self.num_points 
        self.t_waypoints = np.linspace(0, self.t_total, self.num_points)

        self.spline_x = CubicSpline(self.t_waypoints, waypoints[:, 0])
        self.spline_y = CubicSpline(self.t_waypoints, waypoints[:, 1])
        self.spline_z = CubicSpline(self.t_waypoints, waypoints[:, 2])

    def trajectory(self, t: float) -> NDArray[np.floating]:
        """返回时间 t 时刻的轨迹点"""
        x = self.spline_x(t)
        y = self.spline_y(t)
        z = self.spline_z(t)
        return np.array([x, y, z])
    
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict = None) -> NDArray[np.floating]:
        """主控制函数：计算动作"""
        self.actual_trajectory.append(obs["pos"].copy())
        self.init_waypoints, curr_ind = self._augment_gate_waypoints(obs['pos'], obs["gates_pos"], obs['gates_quat'], obs['obstacles_pos'], obs['target_gate'])
        self.planned_waypoints = self.init_waypoints[curr_ind:]
        self.replan(self.planned_waypoints)
        
        if self._tick % 50 == 0:
            t_sampled = np.linspace(0, self.spline_x.x[-1], num=100)  # 采样 100 个时间点
            smooth_points = np.vstack((self.spline_x(t_sampled), self.spline_y(t_sampled), self.spline_z(t_sampled))).T
            self.plot_waypoints(self.init_waypoints, smooth_points, np.array(self.actual_trajectory), self._tick)

        drone_pos = obs["pos"]
        drone_vel = obs["vel"]
        t_now = self.control_tick / self._freq
        self._tick += 1
        self.control_tick += 1
        target_pos = self.trajectory(t_now)

        # 获取障碍数据
        obstacles_pos = obs["obstacles_pos"]
        obstacles_visited = obs["obstacles_visited"]

        # 简单 P 控制器（速度指令）
        kp = 1.5
        vel_cmd = kp * (target_pos - drone_pos)

        # 限速处理
        max_speed = 1.5
        speed = np.linalg.norm(vel_cmd)
        if speed > max_speed:
            vel_cmd = vel_cmd / speed * max_speed

        acc_cmd = np.zeros(3)  # 不使用加速度控制（可后续扩展）

        # 姿态控制参数（可扩展加入航向角）
        yaw = 0.0
        rrate = 0.0
        prate = 0.0
        yrate = 0.0

        # 拼接动作向量 [pos(3) + vel(3) + acc(3) + yaw + rrate + prate + yrate]
        action = np.concatenate((drone_pos, vel_cmd, acc_cmd, [yaw, rrate, prate, yrate]), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        return self._finished
