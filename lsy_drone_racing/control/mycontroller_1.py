from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
from numpy.typing import NDArray


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

    def _augment_gate_waypoints(self,
                            drone_pos: np.ndarray,
                            gates_pos: NDArray[np.floating], 
                            gates_quat: NDArray[np.floating],
                            obstacles_pos: NDArray[np.floating],
                            target_gate_idx: int) -> NDArray[np.floating]:  # 新增目标门参数
        """根据当前target_gate生成穿门轨迹"""
        
        augmented_waypoints = [drone_pos.copy()]
        remaining_gates = gates_pos[target_gate_idx:]  # 只处理未完成的门
        for i in range(len(remaining_gates)):
            current_gate = remaining_gates[i]
            
            # 动态计算门朝向
            if i == 0:
                # 当前目标门：朝向无人机来向
                approach_vector = drone_pos - current_gate
            else:
                # 后续门：指向前一个已处理门的后方
                approach_vector = remaining_gates[i-1] - current_gate
            
            # 处理门朝向向量
            if np.linalg.norm(approach_vector) > 1e-6:
                gate_forward = approach_vector / np.linalg.norm(approach_vector)
            else:
                gate_forward = R.from_quat(gates_quat[target_gate_idx + i]).apply([1, 0, 0])
        
            pre_gate_1 = current_gate - gate_forward * 4.0  # 门前1米
            pre_gate_2 = current_gate - gate_forward * 2.0  # 门前1米

            post_gate_1 = current_gate + gate_forward * 2.0  # 门后1米
            post_gate_2 = current_gate + gate_forward * 4.0  # 门后1米

            augmented_waypoints.extend([pre_gate_1, pre_gate_2, current_gate, post_gate_1, post_gate_2])
        
        return np.array(augmented_waypoints)

    def _get_gate_forward_vector(self, gate_quat: np.ndarray) -> np.ndarray:
        """获取门的正前方向量"""
        rot = R.from_quat(gate_quat)
        return rot.apply([1, 0, 0])  # 假设门的局部x轴是前方

    def _has_obstacle_between(self, p1: np.ndarray, p2: np.ndarray, 
                            obstacles: np.ndarray, step=0.2) -> bool:
        """检查两点间直线路径是否有障碍物"""
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return False
        
        direction = direction / distance
        steps = int(distance / step) + 1
        
        for i in range(steps + 1):
            check_point = p1 + direction * (i * step)
            for obs in obstacles:
                if np.linalg.norm(check_point - obs) < 0.5:  # 障碍物半径阈值
                    return True
        return False

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
        self.init_waypoints = self._augment_gate_waypoints(obs['pos'], obs["gates_pos"], obs['gates_quat'], obs['obstacles_pos'], obs['target_gate'])
        self.replan(self.init_waypoints)
        drone_pos = obs["pos"]
        drone_vel = obs["vel"]
        t_now = self._tick / self._freq
        self._tick += 1
        target_pos = self.trajectory(t_now)

        # 获取障碍数据
        obstacles_pos = obs["obstacles_pos"]
        obstacles_visited = obs["obstacles_visited"]

        repulsion = np.zeros(3)
        safe_distance = 0.4
        for obs_pos, visited in zip(obstacles_pos, obstacles_visited):
            if not visited:
                diff = drone_pos - obs_pos
                dist = np.linalg.norm(diff)
                if 1e-3 < dist < safe_distance:
                    repulsion += (diff / dist) * (1.0 / dist**2)

        # 把避障作用添加到目标点
        target_pos += repulsion

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
