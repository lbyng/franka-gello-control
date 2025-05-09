import argparse
import os
import random

import numpy as np

from gamify.utils.io import load_npz_keys, write_gif, write_mp4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to dataset")
    parser.add_argument("--include_wrist", default=False, action="store_true")
    parser.add_argument("--gif", default=False, action="store_true")
    parser.add_argument("--mp4", default=False, action="store_true")
    parser.add_argument("--num", type=int, default=10)

    args = parser.parse_args()

    assert os.path.exists(args.path), "Dataset path did not exist."
    os.makedirs(args.path[:-1] + "_gifs", exist_ok=True)

    demos = [f for f in os.listdir(args.path) if f.endswith(".npz")]

    random.shuffle(demos)

    for demo in demos[: args.num]:
        print(demo)
        frames = load_npz_keys(os.path.join(args.path, demo), keys=["obs.agent_image"])["obs.agent_image"]
        frames = frames.transpose((0, 2, 3, 1))
        if args.include_wrist:
            wrist_frames = load_npz_keys(os.path.join(args.path, demo), keys=["obs.wrist_image"])["obs.wrist_image"]
            wrist_frames = wrist_frames.transpose((0, 2, 3, 1))
            frames = np.concatenate((frames, wrist_frames), axis=2)
        if args.gif:
            write_gif(frames, os.path.join(args.path[:-1] + "_gifs", demo.replace(".npz", ".gif")))
        if args.mp4:
            write_mp4(frames, os.path.join(args.path[:-1] + "_gifs", demo.replace(".npz", ".mp4")), fps=20)
