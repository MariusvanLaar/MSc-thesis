#!/bin/sh
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp21_mnist_45
#SBATCH --mem=10GB

seeds=(45)


for S in "${seeds[@]}"; do

	echo "Running with $C cuts"
	
	python main_v5.py \
	--tag "Exp21" \
	--seed $S \
	--epochs 50 \
	--kfolds 5 \
	train \
	--optimizer "cwd" \
	--model "PQC-4C" \
	--dataset "mnist-13" \
	--n-layers 2 \
	--n-blocks 8 \
	--n-qubits 8 \
	--observable "All"
	
	sleep 5
	
done
done 