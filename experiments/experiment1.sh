#!/bin/bash

# experiments to run:
#	1) run pong from scratch for (this is to ensure that
#	this doesn't change things too much)
#		a) 5M timestepts
#		b) 10M timesteps
#		c) 50M timesteps
#	2) Run each environment, using a pretrained model on itself, freezing x layers (where x ranges from 0 to number of layers) 
#	3) Run each environment, using a pretrained model on itself, reinitializing x layers
#	4) Run any combination of environments (how do we deal with the different input and action spaces? 

# Experiment i
# pong trained on 5M timesteps

exp_list=(
	"scratch_krull_50M"
	"scratch_krull_2M"
	"pretrained_boxing_2NF"
	"pretrained_boxing_4NF"
	"pretrained_krull_2NF"
	)

#echo "launching experiment1.0"
#exp=${exp_list[0]}
#mkdir -p ./results/${exp}
#nice -n 5 python ./main.py --game krull --T-max 50000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

echo "launching experiment1.1"
exp=${exp_list[1]}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game krull --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

echo "launching experiment1.2"
exp=${exp_list[2]}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game boxing --model ./models/boxing.pth --freeze-layers 5 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

echo "launching experiment1.3"
exp=${exp_list[3]}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game boxing --model ./models/boxing.pth --freeze-layers 3 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

echo "launching experiment1.4"
exp=${exp_list[4]}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game krull --model ./results/scratch_krull_50M/scratch_krull_50M_model.pth --freeze-layers 5 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

