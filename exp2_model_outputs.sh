#!/bin/sh
# Wdbc test for different model outputs


lrs=(0.01 0.05)
opts=("adam")
models=("PQC-3V" "PQC-3W" "PQC-3X" "PQC-3Y" "PQC-3Z" "PQC-4A")


for R in $(seq 1 10); do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do
for M in "${models[@]}"; do

	python main_v4.py \
	--tag "Exp2-"$R \
	--seed $R \
	--epochs 400 \
	--kfolds 10 \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR \
	--model $M \
	--dataset "wdbc"
		
	echo "Run model $M with $R $LR $OPTIM"
	sleep 5
	
done
done
done 
done 