#!/bin/bash

exp_list=(
	"scratch_kangaroo_50M"
	"pretrained_kangaroo_1NF"
	"pretrained_kangaroo_2NF"
	"pretrained_kangaroo_3NF"
	"pretrained_kangaroo_4NF"
	"pretrained_kangaroo_5NF"
	)

exp=${exp_list[0]}
echo "launching experiment" ${exp} 
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp} 

exp=${exp_list[1]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --model ./results/scratch_kangaroo_50M/scratch_kangaroo_50M_model.pth --freeze-layers 6 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[2]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --model ./results/scratch_kangaroo_50M/scratch_kangaroo_50M_model.pth --freeze-layers 5 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[3]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --model ./results/scratch_kangaroo_50M/scratch_kangaroo_50M_model.pth --freeze-layers 4 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[4]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --model ./results/scratch_kangaroo_50M/scratch_kangaroo_50M_model.pth --freeze-layers 3 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

exp=${exp_list[5]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game kangaroo --model ./results/scratch_kangaroo_50M/scratch_kangaroo_50M_model.pth --freeze-layers 2 --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}
