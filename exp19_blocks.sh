#!/bin/sh
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp19_blocks
#SBATCH --mem=10GB

models=("PQC-4B")

for M in "${models[@]}"; do

	echo "Running model $M with dataset $B blocks"
	
	python main_v5.py \
	--tag "Exp19" \
	--seed 765 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset "ion" \
	--n-layers 2 \
	--n-blocks 4 \
	--observable "First"
	
	sleep 5
	
done
