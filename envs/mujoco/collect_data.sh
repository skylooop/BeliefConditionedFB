#!/bin/bash

for i in {0..19}; do
    python /home/m_bobrin/ZeroShotRL/envs/mujoco/main_sac.py --default_ind=$i  # Replace `your_program` with your actual command
done

echo "All runs completed!"