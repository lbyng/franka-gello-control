"""

Example of using local device directly:
python scripts/gello_control.py --config configs/fr3_nocam.yaml --device /dev/ttyACM0

Example of using remote device (after connecting device on rtcrobot.com)
python scripts/gello_control.py --config configs/fr3_nocam.yaml \
    --remote True --secret_id <SECRET_ID> --secret_key <SECRET_KEY> --room_id <ROOM_ID>

If you'd like to save demos, add a --path argument

"""

import argparse
import copy
import datetime
import filecmp
import io
import os
import shutil
from typing import Any, Dict

import gym
import numpy as np
import yaml

from gamify.controllers.gello import GelloController, GelloControllerRemote, GelloControllerWrapper
from gamify.utils import rotations
from gamify.utils.config import Config

NEW_GYM_API = False if gym.__version__ < "0.26.1" else True

rot_gain = 0.75

gello_workspace_limits = np.array(
    [
        [0.13, 0.4],
        [0.18, -0.18],
        [0.0, 0.45],
    ]
)

franka_workspace_limits = np.array(
    [
        [0.31, 0.68],
        [-0.16, 0.24],
        [0.12, 0.73],
    ]
)


def append(lst, item):
    # This takes in a nested list structure and appends everything from item to the nested list structure.
    # It will require lst to have the complete set of keys -- if keys are in item but not in lst,
    # they will not be appended.
    # print(type(item))
    if isinstance(lst, dict):
        assert isinstance(item, dict)
        for k in item.keys():
            append(lst[k], item[k])
    else:
        lst.append(item)


def _flatten_dict_helper(flat_dict: Dict, value: Any, prefix: str, separator: str = ".") -> None:
    if isinstance(value, (dict, gym.spaces.Dict)):
        for k in value.keys():
            assert isinstance(k, str), "Can only flatten dicts with str keys"
            _flatten_dict_helper(flat_dict, value[k], prefix + separator + k, separator=separator)
    else:
        flat_dict[prefix[1:]] = value


def flatten_dict(d: Dict, separator: str = ".") -> Dict:
    flat_dict = dict()
    _flatten_dict_helper(flat_dict, d, "", separator=separator)
    return flat_dict


def nest_dict(d: Dict, separator: str = ".") -> Dict:
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d


def save_episode(data: Dict, path: str, enforce_length: bool = True) -> None:
    # Flatten the dict for saving as a numpy array.
    data = flatten_dict(data)

    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if isinstance(data[k], np.ndarray) and not data[k].dtype == np.float64:  # Allow float64 carve out.
            continue
        elif isinstance(data[k], list):
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, bool):
                dtype = np.bool_
            elif isinstance(first_value, int):
                dtype = np.int64
            else:
                dtype = None
            data[k] = np.array(data[k], dtype=dtype)
        else:
            raise ValueError("Unsupported type passed to `save_data`.")

    if enforce_length:
        # assert len(set(map(len, data.values()))) == 1, "All data keys must be the same length."
        print((map(len, data.values())))

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def record_episode(
    env,
    controller,
    episode_num,
    args,
):
    done = False

    if args.format == "lightning":
        episode = dict(
            reward=[0.0],
            done=[False],
            discount=[1.0],
            desired_action={"action": [env.action_space.sample()]},
        )

        def init_action_fn(space):
            return [space.sample()]

    else:
        episode = dict(
            reward=[],
            done=[],
            discount=[],
            desired_action={"action": []},
        )

    # Write the obs and action to episode
    episode["obs"] = nest_dict({k: [] for k in flatten_dict(env.observation_space).keys()})
    episode["action"] = copy.deepcopy(episode["desired_action"])

    # Reset the environment
    if NEW_GYM_API:
        obs, info = env.reset()
    else:
        obs = env.reset()
    append(episode, dict(obs=obs))

    # See if we want to use language.
    lang = args.instr if args.instr is not None else input("Language instruction? ")
    lang = None if lang == "" else lang
    if lang is not None:
        episode["language_instruction"] = [lang]

    print("Start episode.")

    _base_ee_pos = obs["state"]["ee_pos"]
    base_franka_quat = obs["state"]["ee_quat"]
    _base_gello_pos, base_gello_rmat, _base_gello_gripper = controller.get_ee_pos_rmat_gripper()
    base_gello_quat = rotations.rmat_to_quat(base_gello_rmat)

    while not done:
        try:
            current_franka_ee_pos = obs["state"]["ee_pos"]
            current_franka_quat = obs["state"]["ee_quat"]
            current_gello_pos, current_gello_rmat, current_gello_gripper = controller.get_ee_pos_rmat_gripper()
            current_gello_quat = rotations.rmat_to_quat(current_gello_rmat)

            target_quat_offset = rotations.quat_diff(current_gello_quat, base_gello_quat)
            target_quat_offset = rotations.axisangle_to_quat(rotations.quat_to_axisangle(target_quat_offset))
            desired_quat = rotations.quat_multiply(target_quat_offset, base_franka_quat)

            desired_pos = current_gello_pos.copy()
            desired_pos = (desired_pos - gello_workspace_limits[:, 0]) / (
                gello_workspace_limits[:, 1] - gello_workspace_limits[:, 0]
            )
            desired_pos = franka_workspace_limits[:, 0] + desired_pos * (
                franka_workspace_limits[:, 1] - franka_workspace_limits[:, 0]
            )

            desired_euler_delta = (
                rotations.orientation_error(
                    rotations.quat_to_rmat(desired_quat), rotations.quat_to_rmat(current_franka_quat)
                )
                * rot_gain
            )
            pose = np.concatenate([desired_pos - current_franka_ee_pos, desired_euler_delta, [current_gello_gripper]])

            if NEW_GYM_API:
                obs, reward, done, terminated, info = env.step(pose, controller_type="OSC_POSE", is_delta=True)
            else:
                obs, reward, done, info = env.step(pose, controller_type="OSC_POSE", is_delta=True)
                terminated = False

            action = pose

            discount = 1.0 - float(terminated)

            # Log all of the different action types, but set action to the one in the config file.
            desired_action = info["desired_action"]
            action = desired_action
            step = dict(
                obs=obs,
                action={"action": list(action)},
                reward=reward,
                done=done,
                discount=discount,
                desired_action={"action": list(desired_action)},
            )
            if lang is not None:
                step["language_instruction"] = lang

            append(episode, step)

        except KeyboardInterrupt:
            print("Finished episode.")
            done = True

    # Store done and reward at the final timestep
    episode["done"][-1] = True
    episode["reward"][-1] = 1.0

    if args.format == "bc":
        # Remove the final observation
        episode["obs"] = episode["obs"][:-1]

    success = input("Success [s] or Failure [f]? ") == "s"

    if success:
        print("Saving episode.")
        ep_len = len(episode["done"])
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{episode_num}_{ep_len}.npz"
        save_episode(episode, os.path.join(args.path, ep_filename), enforce_length=(args.format != "rl"))
        return True
    else:
        print("Discarding episode.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config to use for environment")
    parser.add_argument("--path", type=str, required=True, help="Path to save demonstrations.")
    parser.add_argument("--instr", type=str, default=None, help="Language instruction.")

    parser.add_argument("--device", type=str, default="/dev/ttyACM0", help="Device path")

    parser.add_argument("--remote", type=bool, default=False, help="Use remote controller rather than local device")
    parser.add_argument("--secret_id", type=str, default=None, help="RTC secret ID")
    parser.add_argument("--secret_key", type=str, default=None, help="RTC secret key")
    parser.add_argument("--room_id", type=str, default=None, help="RTC room ID")

    parser.add_argument(
        "--format",
        choices=["lightning", "bc", "rl"],
        default="lightning",
        help="How to format the demos for saving.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config_save_path = os.path.join(args.path, "config.yaml")
    if os.path.exists(config_save_path):
        # Check to make sure that the loaded configs were equal
        assert filecmp.cmp(
            args.config, config_save_path
        ), "Trying to add more demos to a folder with a different config."

    config = Config.load(args.config)

    # Parse the config
    config = config.parse()

    # Get the environment
    env = config.get_train_env_fn()()
    if env is None:
        env = config.get_eval_env_fn()

    if args.remote:
        controller = GelloControllerWrapper(
            GelloControllerRemote(secret_id=args.secret_id, secret_key=args.secret_key, room_id=args.room_id)
        )
    else:
        controller = GelloControllerWrapper(GelloController(device_name="/dev/ttyACM0"))

    os.makedirs(args.path, exist_ok=True)
    # Copy the config to the demo storage location.
    shutil.copy(args.config, os.path.join(args.path, "config.yaml"))

    print("Starting data collection.")

    num_episodes = 0
    while True:
        result = record_episode(env, controller, num_episodes, args)
        if result:
            num_episodes += 1


if __name__ == "__main__":
    main()
