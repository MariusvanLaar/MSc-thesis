#!/bin/sh
# Redoing exp15 with 4B many layers
#SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp15_12
#SBATCH --mem=5GB

models=("PQC-4B")
datasets=("wdbc" "spectf" "ion")
observables=("All" "0,0")

for L in $(seq 3 4); do
for M in "${models[@]}"; do
for D in "${datasets[@]}"; do
for O in "${observables[@]}"; do

	echo "Running model $M with dataset $D $L"
	
	python3 main_v5.py \
	--tag "Exp15" \
	--seed 765 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers $L \
	--n-blocks 2 \
	--observable $O 
	
	sleep 5
	
done
done
done 
done 