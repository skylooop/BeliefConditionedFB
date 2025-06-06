# Belief-Conditioned FB

*This repository contains implementation of Belief-FB/Rotation-FB in JAX presented in paper **Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics***

## Environment Setup

All code is written in JAX, so ensure non-conflicting versions of CUDA and latest version of JAX.
Config management is done via Hydra.

1. Conda Environment Setup

	```zsh
	# Create and activate conda environment
	conda create -n BFB python=3.9
	conda activate BFB	
	
	# Install dependencies
	pip install -r requirements.txt
	```

2. Environment Configuration

	```zsh
	# Base project directory
	export PROJECT_DIR=/absolute/path/to/BFB
	
	# Python path configuration
	export SRC_DIR=${PROJECT_DIR}/src
	export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
	```

3. `requirements.txt`: File with Python dependencies

## Code Structure

**Key Components:**
The repository is structured as follows:

1. `agents/`: Folder containing implementations of the BFB/RFB/FB methods (Section 3.2)
2. `notebooks`: Demonstrations of performance of different algorithms applied to benchmarks from paper (Section 4)
	1. `vanilla_fb_doors.ipynb` (on Rnadomized Doors environment)
	1. `vanilla_fb_4rooms.ipynb` (Performance of Vanilla Forward-Backward method on static 4Rooms)
	2. `vanilla_fb_4rooms_randomized.ipynb` (Interference problem illustrated in Section 3.1 on Randomized 4Rooms)

	3. `dynamics_fb_4rooms.ipynb` (Implementation of BFB from Section 3.2 on Randomized 4Rooms)
	4. `dynamics_fb_doors_debug.ipynb` (Visualization of policy encoded space on custom Randomized Doors and RFB, Section 3.3)
	5. `dynamics_fb_4rooms_rot.ipynb` (Performance of RFB on Randomized 4Rooms)


```
├── README.md              
├── requirements.txt       # Libraries dependencies
├── agents/               # Main folder, contating implementations of various methods
	├── dynamics_fb.py # Implementation of BFB from paper
	├── dynamics_rfb.py # Implementation of RFB from paper
├── configs/                   # Config files used for experiments
│   ├── agent    # hyperparameters for each method
│   ├── env         # configs for different benchmarks
│   └── experiment      # Preconfigured config files, each defines an experiment
|	└── entry.yaml # main entry point for hydra
└── data_gen_scripts/ # folder used to generate behavior policies for randomized Pointmass
    ├── datasets/
    │   ├── commands.sh              
    │   └── generate_locomaze.py # Define type of policy, save path, episode length etc.
    ├── envs/ # folder with implmenetations of all benchmarks used in paper
    ├── notebooks/ # Interactive visualizations & training of different agents
    └── utils/
        └── datasets.py     # Contains main class for Dataset creation from numpy array
		└── evaluation.py # Different evaluation procedures based on the bench
		└── flax_utils.py # Flexible and simple extension of flax TrainState
		└── networks.py # Implementations of different network architectures
		└── transformer_nets.py # Implementation of permutation invariant Transformer encoder
```

## Pointmass
For generating random pointmass layouts, run `envs/ogbench/generate_maze_maps.py` and set `num_maps` to any number of grid layouts. The resulting dictionary of grid types will be saved as `pointmaze-maps.npy` file. 

To generate dataset, run `data_gen_scripts/generate locomaze.py` for each layout.
All visualizations and training for `pointmass` are done in `pointmass_fb.ipynb`.

## AntWind
For AntWind we provide dataset, collected by pretrained SAC for various wind directions, which can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1ub8KdqcjgNLZv1aX6TX5mRM6RUxC8_vi?usp=sharing) and placed in folder `envs/mujoco/data_custom_ant`. Visualizations for transformer context disentanglement is in `ant_wind_dynamics.ipynb`.

## Run
`python main_dynamics_discrete.py` (with Wandb logging). If debug: `python main.py --disable_jit=True` (also disables Wandb logging). Provide path to precollected dataset, saved from corresponding env in notebook (ipynb).

For discrete case, select corresponding config from `configs/experiments` folder and run `python main_dynamics_discrete.py experiment=*config name*`