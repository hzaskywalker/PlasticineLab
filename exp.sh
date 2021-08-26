nohup python -m plb.algorithms.solve --env_name Chopsticks-v1 --algo td3 --num_steps 50000 --exp_name exp_funetune --model_name emd_expert_finetune_encoder --srl --path output --seed 2021 &> 20210823_finetune.txt &
nohup python -m plb.algorithms.solve --env_name Chopsticks-v1 --algo td3 --num_steps 50000 --exp_name exp_raw --model_name emd_expert_encoder --srl --path output --seed 2021 &> 20210823_exp.txt &
nohup python -m plb.algorithms.solve --env_name Chopsticks-v1 --algo td3 --num_steps 50000 --exp_name raw --path output --seed 2021 &> 20210823_raw.txt &
