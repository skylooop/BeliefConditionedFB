# ZeroShotRL

*This repository provides implementations of experiments used in the paper, together with implementations of BeliefFB/RotationFB in JAX*

## Environment Setup

All code is written in JAX, so ensure non-conflicting versions of CUDA. 
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

## Code Structure

**Key Components:**

1. `agents/`: Folder containing implementations of the BFB/RFB/FB methods (Section 3.2)
2. `notebooks`: Demonstrations of performance of different algorithms applied to benchmarks from paper (Section 4)
	1. `vanilla_fb_4rooms.ipynb` (Performance of Vanilla Forward-Backward method on static 4Rooms)
	2. `vanilla_fb_4rooms_randomized.ipynb` (Interference problem illustrated in Section 3.1 on Randomized 4Rooms)

	3. `dynamics_fb_4rooms.ipynb` (Implementation of BFB from Section 3.2 on Randomized 4Rooms)
	4. `dynamics_fb_doors_debug.ipynb` (Visualization of policy encoded space on custom Randomized Doors and RFB, Section 3.3)
	5. `dynamics_fb_4rooms_rot.ipynb` (Performance of RFB on Randomized 4Rooms)

```
├── README.md              
├── requirements.txt       # Libraries dependencies
├── agents/               # Main folder, contating implementations of various methods
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
		└── networks.py # Implementations of different NNs
		└── transformer_nets.py # Implementation of permutation invariant Transformer encoder
```

3. `requirements.txt`: File with Python dependencies


## Run
`python main.py` (with Wandb logging). If debug: `python main.py --disable_jit=True` (also disables Wandb logging)