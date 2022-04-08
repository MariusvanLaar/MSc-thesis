#!/bin/sh
# Synthetic dataset tested with various levels of entanglement in ansatz


reps=(2 3)
models=("PQC-4A" "PQC-4B" "PQC-4C")


for L in $(seq 3 8); do
for R in "${reps[@]}"; do
for M in "${models[@]}"; do

	python main_v4.py \
	--tag "Exp4-"$R \
	--seed $R \
	--epochs 200 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--learning-rate 0.05 \
	--model $M \
	--dataset "synth-4a-10" \
	--n-layers $L \
	--batch-size 30
		
	echo "Run model $M with $R $L"
	sleep 5
	
done
done
done
