#!/bin/sh
# Wdbc test for different model outputs


lrs=(0.01 0.05 0.1)
opts=("adam")
models=("PQC-3V" "PQC-3W" "PQC-3X" "PQC-3Y" "PQC-3Z")


for R in $(seq 1 50); do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do
for M in "${models[@]}"; do

	python main_v3.py \
	--tag "Exp2-"$R \
	--seed $R \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR \
	--model $M \
	--dataset "wdbc" \
	--n-blocks 2
		
	echo "Run model $M with $R $LR $OPTIM"
	sleep 5
	
done
done
done 
done 