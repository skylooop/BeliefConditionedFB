# ZeroShotRL

*This repository provides implementations of different experiments used for experiments in the papers, and implementation of BeliefFB/RotationFB*

## Environment Setup

1. Conda Environment Setup

	```zsh
	# Create and activate conda environment
	conda create -n BFB python=3.9
	conda activate BFB	
	
	# Install dependencies
	pip install -r requirements.txt
	```

## Run
`python main.py` (with Wandb logging). If debug: `python main.py --disable_jit=True` (also disables Wandb logging)