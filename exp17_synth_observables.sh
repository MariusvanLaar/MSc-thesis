#!/bin/sh
# PQC output dataset, test for best observable (all or final qubit)


observables=("Final" "All" "0,0")
datasets=("synth-4F" "synth-4A")

for O in "${observables[@]}"; do
for D in "${datasets[@]}"; do

	echo "Running observable $O with dataset $D"
	
	python main_v5.py \
	--tag "Exp17" \
	--seed 1235 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4A" \
	--dataset $D \
	--n-layers 2 \
	--n-blocks 2 \
	--observable $O
	
	sleep 5
	
done
done
