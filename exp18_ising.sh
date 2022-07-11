#!/bin/sh
# Transverse Ising dataset (10 spins), test for benefit of entanglement
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ising
#SBATCH --mem=6GB

models=("PQC-4E" "PQC-4Z")


for M in "${models[@]}"; do
#for Q in $(seq 6 7); do

	echo "Running model $M with $Q qubits"
	
	python main_v5.py \
	--tag "Exp18" \
	--seed 765 \
	--epochs 500 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset "ising-10" \
	--n-layers 2 \
	--n-blocks 2 \
	--n-qubits 1 \
	--observable "0,0"
	
	sleep 5
	
done
#done
