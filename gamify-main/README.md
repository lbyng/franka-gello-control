
## Quick Start

### Commands on Desktop

Clone the repo

```
git clone git@github.com:smirchan/gamify.git
```

Initialize and update submodules
```
git submodule update --init --recursive
```
Create a conda environment

```
conda create -n gamify python=3.11
```
Activate environment and install package·s·
```
conda activate gamify

# Install package as editable
pip install -e .

# Install dependencies·
conda install numpy scipy matplotlib pyyaml
pip install diffusers mujoco "gym==0.26.2" h5py pandas imageio opencv-contrib-python pybullet pyrealsense2 httpx pyserial==3.5 hydra-core tensorboard wandb rich pre-commit

# Install torch; can change for whatever CUDA version is needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install deoxys dependencies (should be already cloned as a submodule under third_party/)
cd third_party
pip install -U -r deoxys_control/deoxys/requirements.txt
pip install -e deoxys_control/deoxys

# Install RTC SDK
cd ../rtc_sdk
pip install -e .
cd ..

```

Install pre-commit hooks:

```
pip install pre-commit
pre-commit install
```

### Deoxys Installation on Control PC (e.g. NUC)
Follow the instructions here: https://github.com/Stanford-ILIAD/deoxys_control - specifically the Installation of Codebase (NUC Server) section. You will need to set the correct IP addresses in `deoxys/config/iliad_nuc.yml` (and make the same changes in the the same file in the copy of `deoxys` on the workstation)

### Starting Deoxys Server
To start the Deoxys server on the NUC, cd into `deoxys_control/deoxys` and run the following, e.g. in tmux windows:
```
./auto_scripts/auto_arm.sh config/iliad_nuc.yml
```
```
python auto_scripts/robotiq_gripper.py --cfg iliad_nuc.yml
```

### Controlling Franka with GELLO

Connect GELLO to desktop. If necessary, set permissions to USB port where GELLO is connected
```
sudo chmod 777 /dev/ttyACM0
```

Can run the following visualization to test that GELLO is working:
```
python gamify/controllers/gello.py
```

Control the Franka with the GELLO (using local device directly)
```
python scripts/gello_control.py --config configs/fr3_nocam.yaml --device /dev/ttyACM0
```

Control the Franka using remote GELLO device (after connecting device on rtcrobot.com)
```
python scripts/gello_control.py --config configs/fr3_nocam.yaml \
    --remote True --secret_id <SECRET_ID> --secret_key <SECRET_KEY> --room_id <ROOM_ID>
```
