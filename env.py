from typing import Any, cast

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class Go2Env(VecEnv):
    def __init__(
        self,
        num_envs: int,
        show_viewer: bool,
        env_cfg: dict[str, Any],
    ) -> None:
        self.num_envs = num_envs
        self.obs_shape = (45,)  # 共计 45 个观测量
        self.num_actions = 12  # 共计 12 个自由度
        self.num_commands = 3  # 给出 2 个线速度和 1 个角速度指令
        self.dt = 0.02  # 控制频率 50 hz
        self.max_episode_length = int(20.0 / self.dt)  # 一个 episode 取 20 秒
        self.device = cast(torch.device, gs.device)

        self.cfg = env_cfg
        friction_ratio_range: list[float] = self.cfg["friction_ratio_range"]
        mass_shift_range: list[float] = self.cfg["mass_shift_range"]
        com_shift_range: list[float] = self.cfg["com_shift_range"]
        commands_range: list[tuple[float, ...]] = list(
            zip(
                self.cfg["lin_vel_x_range"],
                self.cfg["lin_vel_y_range"],
                self.cfg["ang_vel_range"],
            )
        )
        self.commands_upper = torch.tensor(
            commands_range[1], dtype=gs.tc_float, device=self.device
        )
        self.commands_lower = torch.tensor(
            commands_range[0], dtype=gs.tc_float, device=self.device
        )

        self.obs = torch.empty(
            (self.num_envs, *self.obs_shape),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,),
            dtype=gs.tc_int,
            device=self.device,
        )

        init_base_pos = [0.0, 0.0, 0.42]
        init_base_quat = [1.0, 0.0, 0.0, 0.0]
        init_dof_pos_map = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=10,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,  # 禁止同一实体的不同部分发生碰撞
                tolerance=1e-6,  # 收敛容差
                max_collision_pairs=100,  # 当实际碰撞数大于此，则忽略
                batch_dofs_info=True,
                batch_links_info=True,
                constraint_timeconst=0.02,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        # self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file="urdf/plane/plane.urdf",
        #         fixed=True,
        #     )
        # )
        self.scene.add_entity(
            morph=gs.morphs.Terrain(
                pos=(-12.0, -12.0, 0.0),
                randomize=True,
                n_subterrains=(1, 1),
                subterrain_size=(24.0, 24.0),
                horizontal_scale=0.25,
                vertical_scale=0.005,
                subterrain_types="random_uniform_terrain",
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": -0.03,
                        "max_height": 0.03,
                        "step": 0.01,
                        "downsampled_scale": 0.5,
                    }
                },
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=init_base_pos,
                quat=init_base_quat,
            )
        )
        self.scene.build(n_envs=self.num_envs)
        # physics domain randomization
        self.robot.set_friction_ratio(
            friction_ratio=(friction_ratio_range[1] - friction_ratio_range[0])
            * torch.rand(self.num_envs, self.robot.n_links)
            + friction_ratio_range[0],
            links_idx_local=range(self.robot.n_links),
        )
        self.robot.set_mass_shift(
            mass_shift=(mass_shift_range[1] - mass_shift_range[0])
            * torch.rand(self.num_envs, self.robot.n_links)
            + mass_shift_range[0],
            links_idx_local=range(self.robot.n_links),
        )
        self.robot.set_COM_shift(
            com_shift=(com_shift_range[1] - com_shift_range[0])
            * torch.rand(self.num_envs, self.robot.n_links, 3)
            + com_shift_range[0],
            links_idx_local=range(self.robot.n_links),
        )
        # joints 中包含 root joint 和所有 12 个自由度
        self.motors_dof_idx = torch.tensor(
            [joint.dof_start for joint in self.robot.joints[1:]],
            dtype=gs.tc_int,
            device=self.device,
        )
        # self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self.robot.set_dofs_kp([20.0] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([0.5] * self.num_actions, self.motors_dof_idx)
        self.init_base_pos = torch.tensor(
            init_base_pos,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.init_base_quat = torch.tensor(
            init_base_quat,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.inv_init_base_quat = inv_quat(self.init_base_quat)
        self.init_projected_gravity = cast(
            torch.Tensor,
            transform_by_quat(
                torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.device),
                self.inv_init_base_quat,
            ),
        )
        self.init_dof_pos = torch.tensor(
            [init_dof_pos_map[joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.init_qpos = torch.cat(
            [self.init_base_pos, self.init_base_quat, self.init_dof_pos]
        )
        self.base_pos = torch.empty(
            (self.num_envs, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.base_quat = torch.empty(
            (self.num_envs, 4),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.base_lin_vel = torch.empty(
            (self.num_envs, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.base_ang_vel = torch.empty(
            (self.num_envs, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.projected_gravity = torch.empty(
            (self.num_envs, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.dof_pos = torch.empty(
            (self.num_envs, self.num_actions),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.dof_vel = torch.empty(
            (self.num_envs, self.num_actions),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.commands = torch.empty(
            (self.num_envs, self.num_commands),
            dtype=gs.tc_float,
            device=self.device,
        )

    def _resample_commands(self, envs_idx: None | torch.Tensor) -> None:
        commands = (self.commands_upper - self.commands_lower) * torch.rand(
            size=(self.num_envs, self.num_commands),
            dtype=gs.tc_float,
            device=self.device,
        ) + self.commands_lower
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.actions = torch.clip(actions, -100.0, 100.0)
        exec_actions = self.last_actions
        target_dof_pos = exec_actions * 0.25 + self.init_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        base_euler = cast(
            torch.Tensor,
            quat_to_xyz(
                transform_quat_by_quat(self.inv_init_base_quat, self.base_quat),
                rpy=True,
                degrees=True,
            ),
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = cast(
            torch.Tensor, transform_by_quat(self.robot.get_vel(), inv_base_quat)
        )
        self.base_ang_vel = cast(
            torch.Tensor, transform_by_quat(self.robot.get_ang(), inv_base_quat)
        )
        self.projected_gravity = cast(
            torch.Tensor,
            transform_by_quat(
                torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.device),
                inv_base_quat,
            ),
        )
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        rewards_tracking_lin_vel = self._reward_tracking_lin_vel() * 1.0
        rewards_tracking_ang_vel = self._reward_tracking_ang_vel() * 0.2
        rewards_lin_vel_z = self._reward_lin_vel_z() * -1.0
        rewards_action_rate = self._reward_action_rate() * -0.005
        rewards_similar_to_default = self._reward_similar_to_default() * -0.1
        rewards_base_height = self._reward_base_height() * -50.0
        rewards = (
            rewards_tracking_lin_vel
            + rewards_tracking_ang_vel
            + rewards_lin_vel_z
            + rewards_action_rate
            + rewards_similar_to_default
            + rewards_base_height
        )

        self._resample_commands((self.episode_length_buf % int(4.0 / self.dt)) == 0.0)

        time_outs = self.episode_length_buf > self.max_episode_length
        dones = time_outs | (torch.abs(base_euler[:, 0]) > 10.0)
        dones |= torch.abs(base_euler[:, 1]) > 10.0
        self._reset(dones)

        self._update_observations()

        self.last_actions.copy_(self.actions)

        return (
            TensorDict({"obs": self.obs}),
            rewards,
            dones,
            {
                "time_outs": time_outs.float(),
                "log": {
                    "rewards_tracking_lin_vel": rewards_tracking_lin_vel.mean(),
                    "rewards_tracking_ang_vel": rewards_tracking_ang_vel.mean(),
                    "rewards_lin_vel_z": rewards_lin_vel_z.mean(),
                    "rewards_action_rate": rewards_action_rate.mean(),
                    "rewards_similar_to_default": rewards_similar_to_default.mean(),
                    "rewards_base_height": rewards_base_height.mean(),
                },
            },
        )

    def reset(self) -> TensorDict:
        self._reset()
        self._update_observations()
        return TensorDict({"obs": self.obs})

    def get_observations(self) -> TensorDict:
        return TensorDict({"obs": self.obs})

    def _reset(self, dones: None | torch.Tensor = None) -> None:
        self.robot.set_qpos(
            self.init_qpos,
            envs_idx=dones,
            zero_velocity=True,
            skip_forward=True,
        )

        if dones is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()

        else:
            torch.where(
                dones[:, None],
                self.init_base_pos,
                self.base_pos,
                out=self.base_pos,
            )
            torch.where(
                dones[:, None],
                self.init_base_quat,
                self.base_quat,
                out=self.base_quat,
            )
            self.base_lin_vel.masked_fill_(dones[:, None], 0.0)
            self.base_ang_vel.masked_fill_(dones[:, None], 0.0)
            torch.where(
                dones[:, None],
                self.init_projected_gravity,
                self.projected_gravity,
                out=self.projected_gravity,
            )
            torch.where(
                dones[:, None],
                self.init_dof_pos,
                self.dof_pos,
                out=self.dof_pos,
            )
            self.dof_vel.masked_fill_(dones[:, None], 0.0)
            self.actions.masked_fill_(dones[:, None], 0.0)
            self.last_actions.masked_fill_(dones[:, None], 0.0)
            self.episode_length_buf.masked_fill_(dones, 0)

        self._resample_commands(dones)

    def _update_observations(self) -> None:
        # self.obs = torch.cat(
        #     [
        #         self.actions,
        #         self.base_ang_vel * 0.25,
        #         self.projected_gravity,
        #         (self.dof_pos - self.init_dof_pos) * 1.0,
        #         self.dof_vel * 0.05,
        #         self.commands[:, :2] * 2.0,
        #         self.commands[:, 2:] * 0.25,
        #     ],
        #     dim=-1,
        # )
        self.obs = torch.cat(
            [
                self.base_ang_vel,
                self.projected_gravity,
                self.commands,
                (self.dof_pos - self.init_dof_pos),
                self.dof_vel,
                self.actions,
            ],
            dim=-1,
        )

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.init_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - 0.3)
