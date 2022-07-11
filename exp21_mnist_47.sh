#!/bin/sh
#SBATCH --time=8-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp21_mnist_47

seeds=(47)


for S in "${seeds[@]}"; do

	echo "Running with $C cuts"
	
	python main_v5.py \
	--tag "Exp21" \
	--seed $S \
	--epochs 20 \
	--kfolds 5 \
	train \
	--optimizer "cwd" \
	--batch-size 256 \
	--model "PQC-4C" \
	--dataset "mnist-13" \
	--n-layers 1 \
	--n-blocks 8 \
	--n-qubits 8 \
	--observable "All"
	
	sleep 5
	
done 