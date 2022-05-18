#!/bin/sh
# PQC output dataset, test for best observable (all or final qubit)


models=("PQC-4S" "PQC-4T")
datasets=("wdbc" "spectf" "ion")

for M in "${models[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running model $M with dataset $D"
	
	python main_v5.py \
	--tag "Exp15" \
	--seed 234 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model $M \
	--dataset $D \
	--n-layers 1 \
	--n-blocks 2 
	
	sleep 5
	
done
done
