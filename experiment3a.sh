#!/bin/bash

# task list: 
# 	1) train pong agents to make sure horizon doesnt affect things
#	2) get pretrained models for games that I cant pull from online 

exp_list=(
    "scratch_pong_2M"  # 0 this is for testing, to ensure horizon does not affect results
    "scratch_pong_5M"  # 1 this is for testing, to ensure horizon does not affect results
    "scratch_berzerk_10M" # 2 round 1 - get a pretrained model for berzerk, krull, and riverraid 
    "scratch_krull_10M" # 3
    "pretrained_berzerk_1NF_3M" # 4 - 0.0 berzerk to berzerk 
    "pretrained_berzerk_3NF_3M" # 5
    "pretrained_berzerk_1R_3M"  # 6 now repeat with fine tuning
    "pretrained_berzerk_3R_3M"  # 7
    "krull_from_berzerk_1NF_3M" # 8 - 0.1 krull from pretrained berzerk model 
    "krull_from_berzerk_3NF_3M" # 9
    "krull_from_berzerk_1R_3M"  # 10 repeat with fine tuning
    "krull_from_berzerk_3R_3M"  # 11
    "berzerk_from_krull_1NF_3M" # 12 - 1.0 berzerk from pretrined krull emodel 
    "berzerk_from_krull_3NF_3M" # 13
    "berzerk_from_krull_1R_3M"  # 14 repeat with fine tuning
    "berzerk_from_krull_3R_3M"  # 15
    "berzerk_from_riverraid_1NF_3M" # 16 - 2.0 berzerk from riverraid
    "berzerk_from_riverraid_3NF_3M" # 17
    "berzerk_from_riverraid_1R_3M"  # 18 repeat with fine tuning
    "berzerk_from_riverraid_3R_3M"  # 19
    "krull_from_riverriad_1NF_3M"   # 20 - 2.1, krull from riverraid
    "pretrained_berzerk_2NF_3M"     # 21
    "pretrained_berzerk_2R_3M"	    # 22
    "krull_from_berzerk_2NF_3M"     # 23
    "krull_from_berzerk_2R_3M"      # 24
    "berzerk_from_krull_2NF_3M"     # 25
    "berzerk_from_krull_2R_3M"      # 26
    "berzerk_from_riverraid_2NF_3M" # 27
    "berzerk_from_riverraid_2R_3M"  # 28
    "krull_from_riverriad_2NF_3M"   # 29
	)

game_berzerk="berzerk"
game_krull="krull"
game_riverraid="riverraid"
game_pong="pong"
berzerk_model_path="./results/scratch_berzerk_10M/scratch_berzerk_10M_model.pth"
krull_model_path="./results/scratch_krull_10M/scratch_krull_10M_model.pth"
riverraid_model_path="./results/scratch_riverraid_10M/scratch_riverriad_10M_model.pth"

rev_num="2"

t_steps=3000000

# # scratch pong 2M
# exp=${exp_list[0]}
# exp=$exp"_"$rev_num
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_pong} --T-max 2000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# # scratch pong 5M
# exp=${exp_list[1]}
# exp=$exp"_"$rev_num
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_pong} --T-max 5000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# # scratch berzerk 10M
# exp=${exp_list[2]}
# exp=$exp"_"$rev_num
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --T-max 10000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# # scratch krull 10M
# exp=${exp_list[3]}
# exp=$exp"_"$rev_num
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_krull} --T-max 10000000 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# pretrained_berzerk_1NF_3M
exp=${exp_list[4]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# "pretrained_berzerk_3NF_3M" # 5
exp=${exp_list[5]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "pretrained_berzerk_1R_3M"  # 6 now repeat with fine tuning
exp=${exp_list[6]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "pretrained_berzerk_3R_3M"  # 7
exp=${exp_list[7]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "krull_from_berzerk_1NF_3M" # 8 - 0.1 krull from pretrained berzerk model 
exp=${exp_list[8]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "krull_from_berzerk_3NF_3M" # 9
exp=${exp_list[9]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "krull_from_berzerk_1R_3M"  # 10 repeat with fine tuning
exp=${exp_list[10]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#    "krull_from_berzerk_3R_3M"  # 11
exp=${exp_list[11]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "berzerk_from_krull_1NF_3M" # 12 - 1.0 berzerk from pretrined krull emodel 
exp=${exp_list[12]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

# "berzerk_from_krull_3NF_3M" # 13
exp=${exp_list[13]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "berzerk_from_krull_1R_3M"  # 14 repeat with fine tuning
exp=${exp_list[14]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "berzerk_from_krull_3R_3M"  # 15
exp=${exp_list[15]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "berzerk_from_riverraid_1NF_3M" # 16 - 2.0 berzerk from riverraid
exp=${exp_list[16]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}

#     "berzerk_from_riverraid_3NF_3M" # 17
exp=${exp_list[17]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "berzerk_from_riverraid_1R_3M"  # 18 repeat with fine tuning
exp=${exp_list[18]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 1 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "berzerk_from_riverraid_3R_3M"  # 19
exp=${exp_list[19]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


#     "krull_from_riverriad_1NF_3M"   # 20 - 2.1, krull from riverraid
exp=${exp_list[20]}
exp=$exp"_"$rev_num
echo "launching experiment" ${exp}
mkdir -p ./results/${exp}
nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 4 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "pretrained_berzerk_2NF_3M"     # 21
# exp=${exp_list[21]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "pretrained_berzerk_2R_3M"	    # 22
# exp=${exp_list[22]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "krull_from_berzerk_2NF_3M"     # 23
# exp=${exp_list[23]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "krull_from_berzerk_2R_3M"      # 24
# exp=${exp_list[24]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_krull} --model ${berzerk_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "berzerk_from_krull_2NF_3M"     # 25
# exp=${exp_list[25]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "berzerk_from_krull_2R_3M"      # 26
# exp=${exp_list[26]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${krull_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "berzerk_from_riverraid_2NF_3M" # 27
# exp=${exp_list[27]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "berzerk_from_riverraid_2R_3M"  # 28
# exp=${exp_list[28]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_berzerk} --model ${riverraid_model_path} --T-max ${t_steps} --reinitialize-layers 2 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}


# #     "krull_from_riverriad_2NF_3M"   # 29
# exp=${exp_list[29]}
# echo "launching experiment" ${exp}
# mkdir -p ./results/${exp}
# nice -n 5 python ./main.py --game ${game_krull} --model ${riverraid_model_path} --T-max ${t_steps} --freeze-layers 3 --experiment ${exp} --saved-model-path ./results/${exp} --plots-path ./results/${exp} --data-save-path ./results/${exp}
