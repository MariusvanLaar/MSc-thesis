#!/bin/sh
# First hyperparameter optimization on breast cancer dataset


lrs=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
opts=("adam" "spsa" "lbfgs")
reps=("1" "2" "3" "4" "5")

for R in "${reps[@]}"; do
for LR in "${lrs[@]}"; do
for OPTIM in "${opts[@]}"; do

	python main_v2.py \
	--tag "Exp1-"$R \
	train \
	--optimizer $OPTIM \
	--learning-rate $LR 
		
	echo "Run model with $R $LR $OPTIM"
	sleep 5
	
done
done
done 