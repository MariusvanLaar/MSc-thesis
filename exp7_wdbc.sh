#!/bin/sh
# Ionosphere dataset tested with various ansatze 4A and 4C

models=("PQC-4C" "PQC-4D" "PQC-4E")
seed=334

for L in $(seq 7 8); do
for M in "${models[@]}"; do

	echo "Running model $M with $L"
	
	python main_v4.py \
	--tag "Exp7-"$seed \
	--seed $seed \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset "wdbc" \
	--n-layers $L \
	--n-blocks 2	

	sleep 5
	
done
done
