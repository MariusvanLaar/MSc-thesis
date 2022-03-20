#!/bin/sh
# Attempt to train a 20 qubit model on the CIFAR08 dataset


lrs=(0.05)
opts=("adam")
reps=("1" "2" "3" "4" "5")
models=("ANN-01")

for R in "${reps[@]}"; do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do
for M in "${models[@]}"; do

	python main_v3.py \
	--tag "Exp2-"$R \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR \
	--model $M \
	--dataset "CIFAR-PCA-08" \
	--n-blocks 4
		
	echo "Run model $M with $R $LR $OPTIM"
	sleep 5
	
done
done
done 
done 