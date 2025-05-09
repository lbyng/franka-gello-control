# Make sure we have the conda environment set up.
CONDA_PATH=/scr/suvir/miniconda3/bin/activate
ENV_NAME=gamify
REPO_PATH=/scr/suvir/gamify
USE_MUJOCO_PY=false # For using mujoco py
WANDB_API_KEY="" # If you want to use wandb, set this to your API key.
REMOTE_REPO_PATH="suvir@scdt.stanford.edu:/iliad/u/suvir/gamify" # Set the path of a remote copy of the repo if you want to use sync script, e.g. user@server:~/android
SCRATCH_PATH="/scr/suvir/gamify"
RICH_PRINT=true
MATPLOTLIB_MODE=tkagg

export RESEARCH_LIGHTNING_REPO_PATH=$REPO_PATH
export RESEARCH_LIGHTNING_REMOTE_REPO_PATH=$REMOTE_REPO_PATH
export RESEARCH_LIGHTNING_SCRATCH_PATH=$SCRATCH_PATH
export RESEARCH_LIGHTNING_RICH_PRINT=$RICH_PRINT
export RESEARCH_LIGHTNING_MATPLOTLIB_MODE=$MATPLOTLIB_MODE
export HF_DATASETS_CACHE=$REPO_PATH
export WANDB_API_KEY=$WANDB_API_KEY

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME
cd $REPO_PATH
# unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if $WANDB_API_KEY; then
    export WANDB_API_KEY=$WANDB_API_KEY
fi

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    if [ -d "/usr/lib/nvidia" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-11.8" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.8/bin:$PATH
    elif [ -d "/usr/local/cuda-11.7" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.7/bin:$PATH
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA. Compatibility should be verified. torch.compile requires >= 11.7"
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    export MUJOCO_GL="egl"
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
    export MUJOCO_GL="osmesa" # glfw doesn't support headless rendering
fi

export D4RL_SUPPRESS_IMPORT_ERROR=1
