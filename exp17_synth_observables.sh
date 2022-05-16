#!/bin/sh
# PQC output dataset, test for best observable (all or final qubit)


models=("PQC-4A" "PQC-4AA")
datasets=("synth-4A" "synth-4AA")

for D in "${datasets[@]}"; do
for M in "${models[@]}"; do

	echo "Running model $M with $D"
	
	python main_v4.py \
	--tag "Exp17" \
	--seed 1234 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers 4 \
	--n-blocks 2	
	
	sleep 5
	
done
done
