#!/bin/sh
# Hyperparameter optimization on breast cancer dataset


lrs=(0.01 0.05 0.1 0.5 1)
opts=("cma")
reps=("1" "2" "3" "4" "5")
models=("PQC-1A" "PQC-1Y" "PQC-2A" "PQC-2Y" "PQC-3B" "PQC-3D")

for R in "${reps[@]}"; do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do
for M in "${models[@]}"; do

	python main_v2.py \
	--tag "Exp1-"$R \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR \
	--model $M
		
	echo "Run model $M with $R $LR $OPTIM"
	sleep 5
	
done
done
done 
done 