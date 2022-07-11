#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp20_wdbc
#SBATCH --mem=10GB

echo "Running with $C cuts"

python main_v5.py \
--tag "Exp20-test" \
--seed 145 \
--epochs 150 \
--kfolds 2 \
train \
--optimizer "adam" \
--model "PQC-52" \
--dataset "wdbc" \
--n-layers 8 \
--n-blocks 2 \
--observable "First"
