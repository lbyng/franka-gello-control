import datetime
import os
import pickle
import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def make_dir(path, overwrite=False, overwrite_temp=False):
    if os.path.exists(path) and (overwrite or (overwrite_temp and "temp" in path)):
        shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)


def write_config(config, path, filename):
    OmegaConf.save(config, Path(path) / filename)


def load_config(filename, path, overrides=None):
    path = Path(path).expanduser().resolve()
    with initialize_config_dir(str(path), version_base="1.3"):
        if overrides is None:
            config = compose(config_name=filename)
        else:
            config = compose(config_name=filename, overrides=overrides)
        return config


def config_to_dict(config):
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def config_from_dict(d):
    return OmegaConf.create(d)


def config_from_string(s):
    return OmegaConf.create(s)


def write_checkpoint(model, optimizer, iteration, cfg, filename):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": cfg,
    }
    torch.save(state_dict, filename)


def load_checkpoint(model_init, optimizer_init, filename, device):
    state_dict = torch.load(filename, map_location=device)
    model_init.load_state_dict(state_dict["model"])
    model_init.to(device)
    model_init.eval()
    if optimizer_init is not None:
        optimizer_init.load_state_dict(state_dict["optimizer"])
    iteration = state_dict["iteration"]
    return model_init, optimizer_init, iteration


def read_pickle(filename):
    if not str(filename).endswith(".pkl"):
        filename = f"{filename}.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, filename):
    if not str(filename).endswith(".pkl"):
        filename = f"{filename}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_npz(filename):
    if not str(filename).endswith(".npz"):
        filename = f"{filename}.npz"
    x = np.load(filename)
    data = {}
    for item in x.files:
        data[item] = x[item]
    return data


def load_npz_keys(filename, keys):
    if not str(filename).endswith(".npz"):
        filename = f"{filename}.npz"
    x = np.load(filename)
    data = {}
    for item in x.files:
        if item in keys:
            data[item] = x[item]
    return data


def write_npz(data, filename):
    if not str(filename.endswith(".npz")):
        filename = f"{filename}.npz"
    np.savez(filename, **data)


def write_png(arr, filename):
    if not str(filename.endswith(".png")):
        filename = f"{filename}.png"
    Image.fromarray(arr).save(filename)


def read_npy(filename):
    if not str(filename).endswith(".npy"):
        filename = f"{filename}.npy"
    return np.load(filename)


def write_npy(arr, filename):
    if not str(filename).endswith(".npy"):
        filename = f"{filename}.npy"
    np.save(filename, arr)


def write_gif(frames, filename):
    if not str(filename).endswith(".gif"):
        filename = f"{filename}.gif"
    imageio.mimsave(filename, frames)


def write_mp4(frames, filename, fps=10):
    if not str(filename).endswith(".mp4"):
        filename = f"{filename}.mp4"
    writer = imageio.get_writer(filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def display_gif(filename):
    import base64

    from IPython import display

    if not str(filename).endswith(".gif"):
        filename = f"{filename}.gif"
    with open(filename, "rb") as fd:
        b64 = base64.b64encode(fd.read()).decode("ascii")
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')


def display_pointclouds(xyz_list):
    ax = plt.figure().add_subplot(projection="3d")
    for points in xyz_list:
        x, y, z = points.T
        ax.scatter3D(x, y, z, s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal", adjustable="box")
    return ax


def display_traj(xyz, elev=None, azim=None, xlim=None, ylim=None, zlim=None, axis_off=False):
    ax = plt.figure().add_subplot(projection="3d")
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    if axis_off:
        ax.set_axis_off()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax.plot(x, y, zs=z, zdir="z")
    return ax


def display_trajs(xyz_list):
    ax = plt.figure().add_subplot(projection="3d")
    for xyz in xyz_list:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ax.plot(x, y, zs=z, zdir="z")


def display_rgb(frame, transpose=False, show=False):
    if frame.shape[0] == 3:
        transpose = True
    if transpose:
        plt.imshow(frame.transpose((1, 2, 0)))
    else:
        plt.imshow(frame)
    if show:
        plt.show()


def display_temp_gif(frames):
    ts = timestamp()
    filename = f"temp_{ts}.gif"
    write_gif(frames, filename)
    return display_gif(filename)
