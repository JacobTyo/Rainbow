#!/bin/bash

exp_list=(
	"scratch_riverraid_50M"
	"pretrained_riverraid_1NF"
	"pretrained_riverraid_2NF"
	"pretrained_riverraid_3NF"
	"pretrained_riverraid_4NF"
	"pretrained_riverraid_5NF"
	)

exp=${exp_list[0]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[1]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --model ./results/scratch_riverraid_50M/scratch_riverraid_50M_model.pth --freeze-layers 6 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[2]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --model ./results/scratch_riverraid_50M/scratch_riverraid_50M_model.pth --freeze-layers 5 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[3]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --model ./results/scratch_riverraid_50M/scratch_riverraid_50M_model.pth --freeze-layers 4 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[4]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --model ./results/scratch_riverraid_50M/scratch_riverraid_50M_model.pth --freeze-layers 3 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[5]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game riverraid --model ./results/scratch_riverraid_50M/scratch_riverraid_50M_model.pth --freeze-layers 2 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}
