#!/bin/sh
# Ionosphere dataset tested with various ansatze 4A and 4C

models=("PQC-4D" "PQC-4E")
seed=333

for L in $(seq 1 16); do
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
