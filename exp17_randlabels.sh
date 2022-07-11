#!/bin/sh
# PQC output dataset, test for best observable (all or final qubit)
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp17
#SBATCH --mem=4GB

seeds=(25 36 47)
observables=("All" "0,0")

for S in "${seeds[@]}"; do
for O in "${observables[@]}"; do

	echo "Running observable $O with dataset $D"
	
	python main_v5.py \
	--tag "Exp17" \
	--seed $S \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4A" \
	--dataset "synth-4F-rand" \
	--n-layers 2 \
	--n-blocks 2 \
	--learning-rate 0.1 \
	--observable $O
	
	sleep 5
	
done
done


