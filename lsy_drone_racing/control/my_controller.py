"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

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
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self.base_waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                # [0.8, 1.0, 0.2],
                [0.45 ,-0.5 , 0.56],#1
                # [0.2, -1.3, 0.65],
                # [1.1, -0.85, 1.1],#2
                # [0.2, 0.5, 0.65],
                # [0.0, 1.2, 0.525],#3
                # [0.0, 1.2, 1.1],
                # [-0.5, 0.0, 1.1],#4
                # [-0.5, -0.5, 1.1],
            ]
        )
        self.waypoints =self.base_waypoints.copy()
        self.t_total = 8 # 轨迹总时长（秒）
        #轨迹生成参数
        self.original_interval = self.t_total / (len(self.base_waypoints))  # 原时间间隔
        self._tick = 0 # 时间步计数器
        self._freq = config.env.freq # 控制频率（Hz）
        self._finished = False # 轨迹是否完成的标志
        self.gate_processed = [0,0,0,0]  # 新状态标志
        self.gate_idx=0

        # 动态参数
        self.obstacle_threshold = 0.3  # 障碍物触发避障的阈值距离（米）
        self.attraction_gain = 0.0     # 目标点吸引力系数
        self.repulsion_gain = 0.05     # 障碍物排斥力系数

        # 初始化状态跟踪变量
        self.current_target_idx = 0
        self.last_replan_time = 0
        self.new_gate_pos = obs["gates_pos"].copy()
        self.prev_gates_pos = obs["gates_pos"].copy()  # 使用copy避免引用问题
        self.prev_obstacles_pos = obs["obstacles_pos"].copy()
        # print("\n=== 初始坐标 ===")
        # print(f"门框初始坐标:\n{self.prev_gates_pos}")
        # print(f"障碍物初始坐标:\n{self.prev_obstacles_pos}")

    def _check_environment_changes(self, obs: dict):
        need_replan = False
        if not np.array_equal(obs["gates_pos"], self.prev_gates_pos):
            # print(f"检测到门框位置变化")
            # print("门框新坐标:\n", obs["gates_pos"])
            self.new_gate_pos = obs["gates_pos"]
            self.prev_gates_pos = obs["gates_pos"]
            need_replan = True
        return need_replan

    def _generate_new_waypoints(self,gate_idx, current_pos: np.ndarray) -> np.ndarray:
        # print("门框索引:",gate_idx)
        # print("下一个门框坐标:",gate_idx,self.new_gate_pos[gate_idx])
        new_waypoints=np.vstack([current_pos, self.new_gate_pos[gate_idx]])
        return new_waypoints

    def compute_control(self, obs: dict, info: dict = None) -> np.ndarray:
        # 获取当前位置
        current_pos = obs["pos"]
        current_v   = obs["vel"]
        gate_idx = self.gate_idx

        if self._check_environment_changes(obs):
            # print('门变化，重新规划：旧航线:\n',self.waypoints)
            self.waypoints = self._generate_new_waypoints(self.gate_idx,obs["pos"])
            # print("新航线:\n", self.waypoints)
            self.t_total=10
            self._tick = 0
            self._finished = False
        if obs["gates_visited"][self.gate_idx] and self.gate_processed[gate_idx]==0:
            # 生成新航点并重置轨迹
            # print('正在通过第',self.gate_idx+1,"个门")
            # print('旧航线',self.waypoints) 
            if  self.gate_idx ==0:
                self.waypoints = np.array(
                [   current_pos,
                    self.new_gate_pos[gate_idx],
                    [0, -1.3, 0.85],
                    [1.1, -0.85, 1.2],#2
                ])
            elif self.gate_idx ==1:
                self.waypoints = np.array(
                [   current_pos,
                    self.new_gate_pos[gate_idx],
                    [0.0, 1.1, 0.525],#3
                ])
            elif self.gate_idx ==2:
                self.waypoints = np.array(
                [   current_pos,
                    self.new_gate_pos[gate_idx],
                    [0.0, 1.2, 1.1],
                    [-0.5, 0.0, 1.1],#4
                    
                ])
            elif self.gate_idx ==3:
                self.waypoints = np.array(
                [   current_pos,
                    [-0.6, -0.8, 1.1]
                ])
            self.gate_processed [self.gate_idx]= 1
            if self.gate_idx<3:
                self.gate_idx=self.gate_idx+1
                # print("新航线:\n", self.waypoints)
            self.t_total=7
        
        # 轨迹计算（使用动态waypoints）
        tau = min(self._tick / self._freq, self.t_total)
        trajectory = CubicSpline(np.linspace(0, self.t_total, len(self.waypoints)), self.waypoints)
        target_pos = trajectory(tau)

        # # 打印状态信息
        # if self._tick % 50 == 0 :
        #     print(
        #         f"时间步 {self._tick}: "
        #         f"位置 = {np.round(obs['pos'], 4)}, "
        #         f"target_pos = {np.round(target_pos, 4)},"
        #         f"gate_processed = {self.gate_processed}"
        #         # f"gate_visited = {obs}"
        #     )

        # 构造目标状态（位置+速度）
        target_state = np.concatenate([target_pos,np.zeros(10)])
        
        # 检查轨迹完成
        if tau >= 2*self.t_total:
            self._finished = False
            
        return target_state.astype(np.float32)
 
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
        self._tick += 1 # 时间步计数器递增
        return self._finished   # 返回是否完成
        """Reset the time step counter."""
        self._tick = 0
