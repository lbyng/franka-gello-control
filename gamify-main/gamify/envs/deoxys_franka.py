import copy
import os
import time
from typing import Dict, List, Optional

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from deoxys import config_root
from deoxys.franka_interface import DEFAULT_RESET_JOINT, FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.yaml_config import YamlConfig

from gamify.envs.sensors import cameras as sensor_cameras
from gamify.envs.task import Task

NEW_GYM_API = False if gym.__version__ < "0.26.1" else True


def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass


class DeoxysFrankaEnv(gym.Env):
    """
    A simple Gym Environment for controlling robots through Deoxys
    """

    FRANKA_DEFAULT_Q_MIN = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
    FRANKA_DEFAULT_Q_MAX = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]

    def __init__(
        self,
        interface_cfg: str = "iliad_nuc.yml",
        controller_type: str = "OSC_POSE",
        # random_init: bool = True,
        control_hz: float = 15,
        gripper_zero_to_one: bool = False,
        snap_gripper: bool = False,
        img_width: int = 128,  # native output size
        img_height: int = 128,
        resize_img_width: Optional[int] = None,  # desired output size
        resize_img_height: Optional[int] = None,
        depth: bool = False,
        cameras: Optional[Dict] = None,
        channels_first: bool = True,
        horizon: Optional[int] = 500,
        q_min: List[int] = FRANKA_DEFAULT_Q_MIN,
        q_max: List[int] = FRANKA_DEFAULT_Q_MAX,
        q_reset: List[int] = DEFAULT_RESET_JOINT,
        robotiq_offset: bool = True,
        display_img: bool = True,
        display_flipped_wrist: bool = True,
    ):
        self.default_controller_type = controller_type

        # Franka Interface
        if interface_cfg[0] != "/":
            config_path = os.path.join(config_root, interface_cfg)
        else:
            config_path = interface_cfg

        # if true, expects 0 = open, 1 = closed, otherwise -1 to 1
        self.gripper_zero_to_one = gripper_zero_to_one
        # if true, gripper_action > half maps to CLOSED, and < half maps to OPEN
        self.snap_gripper = snap_gripper

        # this is the interface through which we can control the robot
        self._config_path = config_path
        self.robot_interface = None
        self.q_min = q_min
        self.q_max = q_max
        self.q_reset = q_reset

        if robotiq_offset:
            # Offset the wrist joint to account for Robotiq
            self.q_reset[-1] -= np.pi / 4.0

        # Add the action space limits.
        self.action_space = self.get_action_space(self.default_controller_type)

        # Construct the observation space
        spaces = dict(state=self.get_observation_space())

        # optionally override the default image sizes.
        if resize_img_width is None:
            resize_img_width = img_width
        if resize_img_height is None:
            resize_img_height = img_height

        self.img_width = img_width
        self.img_height = img_height

        self.resize_img_width = resize_img_width
        self.resize_img_height = resize_img_height

        # Populate camera observations
        self.cameras = cameras if cameras is not None else dict()
        self._depth = depth
        if cameras is not None:
            for name, _camera_params in cameras.items():
                img_shape = (
                    (3, resize_img_height, resize_img_width)
                    if channels_first
                    else (resize_img_height, resize_img_width, 3)
                )
                spaces[name + "_image"] = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
                if depth and self.cameras[name].has_depth:
                    depth_shape = (
                        (1, resize_img_height, resize_img_width)
                        if channels_first
                        else (resize_img_height, resize_img_width, 1)
                    )
                    spaces[name + "_depth"] = gym.spaces.Box(low=0, high=2**16 - 1, shape=depth_shape, dtype=np.uint16)

        self.observation_space = gym.spaces.Dict(spaces)
        # self.random_init = random_init
        self.horizon = horizon
        self._max_episode_steps = horizon  # Added so it looks like we have a gym time limit wrapper.
        self.control_hz = float(control_hz)
        self.channels_first = channels_first
        self._steps = 0

        # a "task" determines rewards
        self._task = None

        self.display_img = display_img
        self.display_flipped_wrist = display_flipped_wrist
        self._img = None
        self._thread = None

        self._last_frames = None
        self._last_frames_unscaled = None

        self._initialized = False

    def _initialize(self):
        self._initialize_robot()
        self._initialize_cameras()
        self._initialized = True

    def _initialize_robot(self):
        self.robot_interface = FrankaInterface(self._config_path, control_freq=self.control_hz)

    def _initialize_cameras(self):
        for name, camera_params in self.cameras.items():
            camera_class, camera_kwargs = camera_params["camera_class"], camera_params["camera_kwargs"]
            self.cameras[name] = vars(sensor_cameras)[camera_class](
                width=self.img_width, height=self.img_height, depth=self._depth, **camera_kwargs
            )

    def compliant(self):
        def _compliant_step():
            self.step(
                action=[*list(self.robot_interface.last_q), -1],
                controller_type="JOINT_IMPEDANCE",
                controller_cfg="compliant-joint-impedance-controller.yml",
            )

        try:
            while True:
                _compliant_step()
        except KeyboardInterrupt:
            pass

    def get_observation_space(self):
        obs_space = {
            "O_T_EE": (16,),
            "O_T_EE_d": (16,),
            "F_T_EE": (16,),
            "F_T_NE": (16,),
            "NE_T_EE": (16,),
            "EE_T_K": (16,),
            "m_ee": (1,),
            "I_ee": (9,),
            "F_x_Cee": (3,),
            "m_load": (1,),
            "I_load": (9,),
            "F_x_Cload": (3,),
            "m_total": (1,),
            "I_total": (9,),
            "F_x_Ctotal": (3,),
            "elbow": (2,),
            "elbow_d": (2,),
            "elbow_c": (2,),
            "delbow_c": (2,),
            "ddelbow_c": (2,),
            "tau_J": (7,),
            "tau_J_d": (7,),
            "dtau_J": (7,),
            "q": (7,),
            "q_d": (7,),
            "dq": (7,),
            "dq_d": (7,),
            "ddq_d": (7,),
            "joint_contact": (7,),
            "cartesian_contact": (6,),
            "joint_collision": (7,),
            "cartesian_collision": (6,),
            "tau_ext_hat_filtered": (7,),
            "O_F_ext_hat_K": (6,),
            "K_F_ext_hat_K": (6,),
            "O_dP_EE_d": (6,),
            "O_T_EE_c": (16,),
            "O_dP_EE_c": (6,),
            "O_ddP_EE_c": (6,),
            "theta": (7,),
            "dtheta": (7,),
            "control_command_success_rate": (1,),
            "time": (1,),
            "gripper_last_action": (1,),
            "gripper_q": (1,),
            "gripper_is_grasped": (1,),
            # our fields
            "ee_pos": (3,),
            "ee_quat": (4,),
            "gripper_pos": (1,),
        }

        for name, shape in obs_space.items():
            obs_space[name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        return gym.spaces.Dict(obs_space)

    def get_action_space(self, controller_type):
        if controller_type in ["OSC_POSE", "OSC_POSITION", "OSC_YAW", "CARTESIAN_VELOCITY"]:
            # delta pose by default
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        if controller_type in ["JOINT_POSITION", "JOINT_IMPEDANCE"]:
            # absolute joints by default
            return gym.spaces.Box(
                low=[*list(self.q_min), -1.0], high=[*list(self.q_max), 1.0], shape=(8,), dtype=np.float32
            )

        raise NotImplementedError(controller_type)

    def _get_frames(self):
        obs_frames = dict()
        obs_frames_unscaled = dict()
        for name, camera in self.cameras.items():
            frames_unscaled = camera.get_frames()
            # resize images if necessary
            if (self.resize_img_width, self.resize_img_height) != (self.img_width, self.img_height):
                frames = {
                    k: cv2.resize(v, (self.resize_img_width, self.resize_img_height), interpolation=cv2.INTER_AREA)
                    for k, v in frames_unscaled.items()
                }
            else:
                frames = copy.deepcopy(frames_unscaled)
            if self.channels_first:
                frames = {k: v.transpose(2, 0, 1) for k, v in frames.items()}
                frames_unscaled = {k: v.transpose(2, 0, 1) for k, v in frames_unscaled.items()}
            obs_frames.update({name + "_" + k: v for k, v in frames.items()})
            obs_frames_unscaled.update({name + "_" + k: v for k, v in frames_unscaled.items()})
        self._last_frames = obs_frames
        self._last_frames_unscaled = obs_frames_unscaled
        return obs_frames

    def _get_obs(self):
        state = self.robot_interface.last_state
        assert state is not None, "Missing state from robot interface!"
        state_dict = dict()
        for key in self.observation_space["state"].keys():
            if key not in ["ee_pos", "ee_quat"] and not key.startswith("gripper"):
                state_dict[key] = getattr(state, key)
                if key == "time":
                    state_dict[key] = state_dict[key].toSec
                state_dict[key] = np.array(state_dict[key])
        state_dict["ee_quat"], state_dict["ee_pos"] = self.robot_interface.last_eef_quat_and_pos
        state_dict["ee_quat"], state_dict["ee_pos"] = (
            state_dict["ee_quat"].reshape(-1),
            state_dict["ee_pos"].reshape(-1),
        )

        last_gripper_action = self.robot_interface.last_gripper_action
        last_gripper_q = self.robot_interface.last_gripper_q
        is_grasped = np.array(self.robot_interface._gripper_state_buffer[-1].is_grasped)
        gripper_max_width = np.array(self.robot_interface._gripper_state_buffer[-1].max_width)

        state_dict["gripper_last_action"] = last_gripper_action
        state_dict["gripper_q"] = last_gripper_q

        # 1 if object grasp detected, 0 otherwise
        state_dict["gripper_is_grasped"] = is_grasped.astype(np.float32)

        # Gripper pos is normalized such that 0 is open and 1 is closed.
        # This is unaffected by the gripper_zero_to_one flag (which is only for actions).
        state_dict["gripper_pos"] = 1 - last_gripper_q / gripper_max_width
        obs = dict(state=state_dict)
        obs.update(self._get_frames())

        return obs

    def step(
        self,
        action,
        controller_type: Optional[str] = None,
        controller_cfg: Optional[str] = None,
        is_delta: Optional[bool] = None,
        policy_id=None,
    ):
        # record the step start time.
        if self._time is None:
            self._time = time.time()

        if controller_type is None:
            controller_type = self.default_controller_type

        if controller_cfg is None:
            controller_cfg = get_default_controller_config(controller_type=controller_type)
        elif isinstance(controller_cfg, str):
            controller_cfg = YamlConfig(
                os.path.join(config_root, controller_cfg),
            ).as_easydict()

        if is_delta is not None:
            controller_cfg.update({"is_delta": is_delta})

        # run an arbitrary controller in deoxys with the given control type, using default params
        self.robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

        # Make sure get_obs gets called at control_hz
        # (This is true assuming get_obs() takes a constant amount of time)
        # end_time = self._time + (1 / self.control_hz)
        # precise_wait(end_time)
        self._time = time.time()

        # get the latest obs after waiting
        obs = self._get_obs()
        self._steps += 1
        terminated = self.horizon is not None and self._steps == self.horizon
        info = dict(discount=1 - float(terminated))

        info["desired_action"] = action
        # denote the task name (a string) in info
        if self._task is not None:
            info["task"] = self._task.name
        else:
            info["task"] = ""

        # compute the reward.
        reward = self.get_reward(obs)

        if self.display_img:
            concat_image = []
            for k, v in obs.items():
                if "_image" in k:
                    if "wrist" in k and self.display_flipped_wrist:
                        concat_image.append(v[:, ::-1, ::-1])
                    else:
                        concat_image.append(v)
            concat_image = np.concatenate(concat_image, axis=1)
            concat_image = concat_image.transpose((1, 2, 0))
            if self._img is None:
                self._img = plt.imshow(concat_image)
            else:
                self._img.set_data(concat_image)
            plt.pause(0.0001)

        if NEW_GYM_API:
            # Note that this is following the Gym 0.26 API for termination.
            return obs, reward, False, terminated, info
        else:
            return obs, reward, terminated, info

    def verify_scene_has_not_changed(self):
        return True

    def reset(self, task: Optional[Task] = None, tolerance=1.3e-3, joint_pos=None, policy_id=None, reset_scene=None):
        if not self._initialized:
            self._initialize()
        # self.controller.reset(randomize=self.random_init)
        # TODO randomization
        if joint_pos is None:
            joint_pos = self.q_reset
        self.robot_interface.goto_joint_position(joint_positions=joint_pos, tolerance=tolerance)

        self._steps = 0
        self._time = None
        obs = self._get_obs()

        # optionally set or reset the task
        self.reset_task(obs, task)

        if NEW_GYM_API:
            return obs, dict()
        else:
            return obs

    def reset_task(self, obs, task: Optional[Task] = None):
        if task is not None:
            self._task = task

        if self._task is not None:
            self._task.reset(self, obs)

    def get_reward(self, obs):
        reward = 0
        if self._task is not None:
            reward = float(self._task.get_reward(self, obs))

        return reward
