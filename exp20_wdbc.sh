#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp20_wdbc
#SBATCH --mem=3GB


for S in $(seq 1112 1115); do
for C in $(seq 0 7); do

	echo "Running with $C cuts"
	
	python main_v5.py \
	--tag "Exp20" \
	--seed $S \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-52" \
	--dataset "wdbc" \
	--n-layers $C \
	--n-blocks 2 \
	--observable "First"
	
	sleep 5
	
done
done 