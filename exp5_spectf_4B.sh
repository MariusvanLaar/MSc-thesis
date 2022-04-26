#!/bin/sh
# SPECTF dataset tested with various levels of entanglement in ansatz PQC-4B


#for L in $(seq 1 4); do
#
#	echo "Running model $M with $L, 2 blocks"
	
#	python main_v4.py \
#	--tag "Exp5-2" \
#	--seed 2 \
#	--epochs 150 \
#	--kfolds 10 \
#	train \
#	--optimizer "adam" \
#	--model "PQC-4B" \
#	--dataset "spectf" \
#	--n-layers $L \
#	--n-blocks 2
		
#	sleep 5
	
#done

for L in $(seq 1 2); do

	echo "Running model with $L, 4 blocks"
	
	python main_v4.py \
	--tag "Exp5-2" \
	--seed 2 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4B" \
	--dataset "spectf" \
	--n-layers $L \
	--n-blocks 4
		
	sleep 5
	
done


echo "Running model with $L, 6 blocks"
	
python main_v4.py \
	--tag "Exp5-2" \
	--seed 2 \
	--epochs 150 \
	--kfolds 10 \
	train \
	--optimizer "adam" \
	--model "PQC-4B" \
	--dataset "spectf" \
	--n-layers 1 \
	--n-blocks 6
		
sleep 5

