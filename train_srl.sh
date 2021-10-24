python -m plb.algorithms.solve --env_name Torus-v1 --algo one_step --batch_size 4 --lr 1e-5 --exp_name torus --model_name finetune_torus_whole &> learn_torus_latent.txt
python -m plb.algorithms.solve --env_name Writer-v1 --algo one_step --batch_size 4 --lr 1e-5 --exp_name writer --model_name finetune_writer_whole &> learn_writer_latent.txt
