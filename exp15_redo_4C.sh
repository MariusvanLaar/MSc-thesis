#!/bin/sh
# Redoing exp15 with "First" observable
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp15_4C

observables=("All" "0,0")
datasets=("wdbc" "spectf" "ion")

for L in $(seq 1 10); do
for O in "${observables[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running obs $O with dataset $D $L"
	
	python3 main_v5.py \
	--tag "Exp15" \
	--seed 765 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4C" \
	--dataset $D \
	--n-layers $L \
	--n-blocks 2 \
	--observable $O
	
	sleep 5
	
done
done
done 