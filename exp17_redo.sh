#!/bin/sh
# PQC output dataset, test for best observable (all or final qubit)
#SBATCH --time=0:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp17

observables=("First")
datasets=("synth-4A")

for O in "${observables[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running observable $O with dataset $D"
	
	python main_v5.py \
	--tag "Exp17" \
	--seed 1235 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4A" \
	--dataset $D \
	--n-layers 2 \
	--n-blocks 2 \
	--observable $O
	
	sleep 5
	
done
done

observables=("First")
datasets=("synth-4F")

for O in "${observables[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running observable $O with dataset $D"
	
	python main_v5.py \
	--tag "Exp17" \
	--seed 1235 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4A" \
	--dataset $D \
	--n-layers 2 \
	--n-blocks 2 \
	--observable $O
	
	sleep 5
	
done
done
