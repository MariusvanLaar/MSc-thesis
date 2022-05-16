#!/bin/sh
# All datasets tested with ansatze 4S and 4T to test effect of datareuploading (compare against exp15 data)

models=("PQC-4T")
datasets=("ion" "spectf" "wdbc")

seed=234

for D in "${datasets[@]}"; do
for L in $(seq 4 7); do
for M in "${models[@]}"; do

	echo "Running model $M with $L $D"
	
	python main_v4.py \
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
