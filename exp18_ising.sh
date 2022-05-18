#!/bin/sh
# Transverse Ising dataset (10 spins), test for benefit of entanglement


models=("PQC-4B" "PQC-4C")
datasets=("ising-10")

for M in "${models[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running model $M with dataset $D"
	
	python main_v5.py \
	--tag "Exp18" \
	--seed 765 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers 2 \
	--n-blocks 2 
	
	sleep 5
	
done
done
