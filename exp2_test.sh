#!/bin/sh
# Wdbc test for different model outputs


lrs=(0.05)
reps=(1)
opts=("adam")
models=("PQC-4A")


#for R in $(seq 1 10); do
for R in "${reps[@]}"; do
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