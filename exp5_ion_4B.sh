#!/bin/sh
# Ionosphere dataset tested with various levels of entanglement in ansatz PQC-4B


reps=(2 3 4 5 6)
models=("PQC-4B")

for B in $(seq 2 2 4); do
for L in $(seq 3 5); do
for R in "${reps[@]}"; do
for M in "${models[@]}"; do

	echo "Running model $M with $R $L $B"
	
	python main_v4.py \
	--tag "Exp5-"$R \
	--seed $R \
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
done
