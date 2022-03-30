#!/bin/sh
# Wdbc test for different model outputs


lrs=(0.01)
opts=("adam")
reps=(2 3)
models=("PQC-4A" "PQC-3W")


for R in "${reps[@]}"; do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do
for M in "${models[@]}"; do

	python main_v4.py \
	--tag "Exp3-"$R \
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