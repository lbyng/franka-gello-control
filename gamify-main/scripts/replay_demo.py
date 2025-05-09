"""
A simple script for replaying a demonstration
"""

import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from gamify.utils.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to demonstration.")
    parser.add_argument("--ref-path", type=str, required=False, default="", help="Path to reference demonstration.")
    parser.add_argument("--wait", type=float, required=False, default=0.001)
    parser.add_argument("--use-robot", default=False, action="store_true", help="Set to True to enable the robot.")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to environment config.")

    args = parser.parse_args()

    assert os.path.exists(args.path), "Demo did not exist."
    demo_dir = os.path.dirname(args.path)
    ref_demo_dir = os.path.dirname(args.ref_path)
    config_path = os.path.join(demo_dir, "config.yaml")

    with open(args.path, "rb") as f:
        data = np.load(f)
        actions = data["desired_action.action"]
        images = np.concatenate((data["obs." + "agent_image"], data["obs." + "wrist_image"]), axis=2)
        is_channels_first = images.shape[-1] != 3

    if args.ref_path != "":
        with open(args.ref_path, "rb") as f:
            ref_data = np.load(f)
            ref_actions = ref_data["action"]
            ref_images = np.concatenate((ref_data["obs." + "agent_image"], ref_data["obs." + "wrist_image"]), axis=2)
            ref_is_channels_first = ref_images.shape[-1] != 3

    if args.config is not None:
        # Load from a config file
        config = Config.load(args.config)
        config = config.parse()
        if args.use_robot:
            env = config.get_train_env_fn()()
            env.reset()

    border_size = 5

    image = images[0]
    if is_channels_first:
        image = image.transpose(1, 2, 0)

    if args.ref_path != "":
        ref_image = ref_images[0]
        if ref_is_channels_first:
            ref_image = ref_image.transpose(1, 2, 0)

        image = np.concatenate((image, ref_image), axis=1)

    display = plt.imshow(image)
    plt.ion()
    plt.show()
    for i in range(min(actions.shape[0], images.shape[0] - 1)):
        if args.use_robot:
            env.step(actions[i + 1], "OSC_POSE")

        image = images[i + 1]
        if is_channels_first:
            image = image.transpose(1, 2, 0)

        if args.ref_path != "":
            ref_image = ref_images[min(i + 1, len(ref_images) - 1)]
            if ref_is_channels_first:
                ref_image = ref_image.transpose(1, 2, 0)

            image = np.concatenate((image, ref_image), axis=1)

        display.set_data(image)
        plt.pause(args.wait)

    print("[robots] Done replaying.")
