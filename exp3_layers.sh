#!/bin/sh
# Wdbc test for different model outputs


lrs=(0.05)
opts=("adam")
reps=(2 3)
models=("PQC-4A")


for L in $(seq 1 6); do
for R in "${reps[@]}"; do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do


	python main_v4.py \
	--tag "Exp3-"$R \
	--seed $R \
	--epochs 300 \
	--kfolds 8 \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR \
	--model "PQC-4A" \
	--dataset "wdbc" \
	--n-layers $L
		
	echo "Run model $M with $R $LR $OPTIM"
	sleep 5
	
done
done
done 
done 