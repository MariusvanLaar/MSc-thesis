#!/bin/sh
# Ionosphere dataset tested with various ansatze 4A for additional gradient data


models=("PQC-4A")
seed=25

for B in $(seq 2 6); do

	echo "Running model 4A with $B blocks"
	
	python main_v4.py \
	--tag "Exp5-"$seed \
	--seed $seed \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4A" \
	--dataset "ion" \
	--n-layers 3 \
	--n-blocks $B	
	
	sleep 5
	
done

