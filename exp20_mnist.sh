#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp20_mnist
#SBATCH --mem=10GB

seeds=(45)


for S in "${seeds[@]}"; do
for C in $(seq 0 7); do

	echo "Running with $C cuts"
	
	python main_v5.py \
	--tag "Exp20" \
	--seed $S \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-52" \
	--dataset "mnist-13" \
	--n-layers $C \
	--n-blocks 8 \
	--n-qubits 8 \
	--observable "First" \
	--learning-rate 1
	
	sleep 5
	
done
done 