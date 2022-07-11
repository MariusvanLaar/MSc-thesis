#!/bin/sh
# Redoing exp15 with "First" observable
# SBATCH --time=06:00:00
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --job-name=exp15

models=("PQC-4B" "PQC-4C")
datasets=("wdbc" "spectf" "ion")

for M in "${models[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running model $M with dataset $D"
	
	python main_v5.py \
	--tag "Exp15" \
	--seed 765 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers 1 \
	--n-blocks 2 \
	--observable "First"
	
	sleep 5
	
done
done