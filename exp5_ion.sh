#!/bin/sh
# Ionosphere dataset tested with various ansatze 4A and 4C

models=("PQC-4B" "PQC-4C" "PQC-4D" "PQC-4E")
datasets=("ion" "spectf" "wdbc")

seed=234

for D in "${datasets[@]}"; do
for L in $(seq 2 4); do
for M in "${models[@]}"; do

	echo "Running model $M with $L $D"
	
	python main_v5.py \
	--tag "Exp15-"$seed \
	--seed $seed \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers $L \
	--n-blocks 2 

	sleep 5
	
done
done
done
