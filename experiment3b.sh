#!/bin/bash

exp_list=(
    "scratch_pong_10M" # 0 this is for testing, to ensure horizon does not affect results
    "scratch_riverraid_10M" # 1 round 1
    "pretrained_krull_1NF_3M"  # 2 - 1.1 we now have pretrained models, run krull to krull
    "pretrained_krull_3NF_3M"  # 3
	"pretrained_krull_1R_3M"   # 4 repeat with fine tuning
    "pretrained_krull_3R_3M"   # 5
    "pretrained_riverraid_1NF_3M" # 6 - 2.2 now with riverraid to riverraid
    "pretrained_riverraid_3NF_3M" # 7
    "pretrained_riverraid_1R_3M"  # 8 repeat with fine tuning
    "pretrained_riverraid_3R_3M"  # 9
    "riverraid_from_berzerk_1NF_3M" # 10 - 0.2 riverraid from pretrined berzerk model
    "riverraid_from_berzerk_3NF_3M" # 11
    "riverraid_from_berzerk_1R_3M"  # 12 repeat with fine tuning
    "riverraid_from_berzerk_3R_3M"  # 13
    "riverraid_from_krull_1NF_3M"   # 14 - 1.2 riverraid from krull pretrined model 
    "riverraid_from_krull_3NF_3M"   # 15
    "riverraid_from_krull_1R_3M"    # 16 repeat with fine tuning
    "riverraid_from_krull_3R_3M"    # 17
    "krull_from_riverriad_3NF_3M"   # 18 krull from riverraid
    "krull_from_riverriad_1R_3M"    # 19 repeat with fine tuning 
    "krull_from_riverriad_3R_3M"    # 20
     "pretrained_krull_2NF_3M"      # 21
    "pretrained_krull_2R_3M"        # 22
    "pretrained_riverraid_2NF_3M"   # 23 
    "pretrained_riverraid_2R_3M"    # 24
    "riverraid_from_berzerk_2NF_3M" # 25
    "riverraid_from_berzerk_2R_3M"  # 26
    "riverraid_from_krull_2NF_3M"   # 27
    "riverraid_from_krull_2R_3M"    # 28
    "krull_from_riverriad_2R_3M"    # 29
	)

game_berzerk="berzerk"
game_krull="krull"
game_riverraid="riverraid"
game_pong="pong"
berzerk_model_path="./results/scratch_berzerk_10M/scratch_berzerk_10M_model.pth"
krull_model_path="./results/scratch_krull_10M/scratch_krull_10M_model.pth"
riverraid_model_path="./results/scratch_riverraid_10M/scratch_riverriad_10M_model.pth"

t_steps=3000000


#    "scratch_pong_10M" # 0 this is for testing, to ensure horizon does not affect results
exp=${exp_list[0]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_pong} --T-max 10000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "scratch_riverraid_10M" # 1 round 1
exp=${exp_list[1]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --T-max 10000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "pretrained_krull_1NF_3M"  # 2 - 1.1 we now have pretrained models, run krull to krull
exp=${exp_list[2]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "pretrained_krull_3NF_3M"  # 3
exp=${exp_list[3]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# "pretrained_krull_1R_3M"   # 4 repeat with fine tuning
exp=${exp_list[4]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "pretrained_krull_3R_3M"   # 5
exp=${exp_list[5]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "pretrained_riverraid_1NF_3M" # 6 - 2.2 now with riverraid to riverraid
exp=${exp_list[6]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_riverraid_3NF_3M" # 7
exp=${exp_list[7]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_riverraid_1R_3M"  # 8 repeat with fine tuning
exp=${exp_list[8]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_riverraid_3R_3M"  # 9
exp=${exp_list[9]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_1NF_3M" # 10 - 0.2 riverraid from pretrined berzerk model
exp=${exp_list[10]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_3NF_3M" # 11
exp=${exp_list[11]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_1R_3M"  # 12 repeat with fine tuning
exp=${exp_list[12]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_3R_3M"  # 13
exp=${exp_list[13]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_1NF_3M"   # 14 - 1.2 riverraid from krull pretrined model 
exp=${exp_list[14]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_3NF_3M"   # 15
exp=${exp_list[15]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_1R_3M"    # 16 repeat with fine tuning
exp=${exp_list[16]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_3R_3M"    # 17
exp=${exp_list[17]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "krull_from_riverriad_3NF_3M"   # 18 krull from riverraid
exp=${exp_list[18]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "krull_from_riverriad_1R_3M"    # 19 repeat with fine tuning 
exp=${exp_list[19]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "krull_from_riverriad_3R_3M"    # 20
exp=${exp_list[20]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "pretrained_krull_2NF_3M"      # 21
exp=${exp_list[21]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_krull_2R_3M"        # 22
exp=${exp_list[22]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_riverraid_2NF_3M"   # 23 
exp=${exp_list[23]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "pretrained_riverraid_2R_3M"    # 24
exp=${exp_list[24]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_2NF_3M" # 25
exp=${exp_list[25]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_berzerk_2R_3M"  # 26
exp=${exp_list[26]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_2NF_3M"   # 27
exp=${exp_list[27]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "riverraid_from_krull_2R_3M"    # 28
exp=${exp_list[28]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_riverraid} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#    "krull_from_riverriad_2R_3M"    # 29
exp=${exp_list[29]}
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}
