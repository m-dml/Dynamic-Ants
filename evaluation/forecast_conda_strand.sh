#!/usr/bin/env bash

#SBATCH --job-name=AI-HERO-Energy_baseline_forecast_conda
#SBATCH --partition=pGPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=./baseline_forecast_conda.txt

export CUDA_CACHE_DISABLE=1

data_dir=/gpfs/work/machnitz
weights_path=/gpfs/work/machnitz/weights/

/gpfs/home/machnitz/miniconda3/envs/hydra/bin/python -u /gpfs/home/machnitz/Dynamic-Ants/forecast.py --save_dir "$PWD" --data_dir ${data_dir} --weights_path ${weights_path}
