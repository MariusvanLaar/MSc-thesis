#!/bin/sh
# ION dataset tested with random cut cnots in ansatz PQC-5A

seeds=(165 167 169)

for S in "${seeds[@]}"; do
for L in $(seq 1 8); do

	echo "Running model $M with $L, $S"
	
	python main_v4.py \
	--tag "Exp6-"$S \
	--seed $S \
	--epochs 150 \
	--kfolds 4 \
	train \
	--optimizer "adam" \
	--model "PQC-5A" \
	--dataset "ion" \
	--n-layers $L \
	--n-blocks 3
		
	sleep 5
	
done
done
