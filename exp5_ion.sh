#!/bin/sh
# Ionosphere dataset tested with various ansatze 4A and 4C

models=("PQC-4A" "PQC-4C")

for B in $(seq 2 2 6); do
for L in $(seq 1 7); do
for M in "${models[@]}"; do

	echo "Running model $M with $L $B"
	
	python main_v4.py \
	--tag "Exp5-2" \
	--seed 2 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset "ion" \
	--n-layers $L \
	--n-blocks $B	

	sleep 5
	
done
done
done
