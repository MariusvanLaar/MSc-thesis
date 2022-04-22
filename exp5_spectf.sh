#!/bin/sh
# Ionosphere dataset tested with various levels of entanglement in ansatz PQC-4B


models=("PQC-4A" "PQC-4C")

for B in $(seq 2 2 6); do
for L in $(seq 1 7); do
for M in "${models[@]}"; do

	echo "Running model $M with $R $L $B"
	
	python main_v4.py \
	--tag "Exp5-0" \
	--seed 1234 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset "spectf" \
	--n-layers $L \
	--n-blocks $B	
	
	sleep 5
	
done
done
done
