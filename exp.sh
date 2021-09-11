#/bin/bash
models=$(ls pretrain_model)
stamp=$(date "+%m%d%H%M%S")
mkdir -p out
mkdir -p out/autoencoder

interpreter="python3"
trainer="plb.algorithms.autoencoder.train_autoencoder"
freezeEncoder="--freeze_encoder"

export CUDA_VISIBLE_DEVICES=0;

for modelFileName in $models
do  modelName=${modelFileName%.pth}
    expName="--exp ${modelName}_$stamp"
    savedModel="--saved_model $modelName"

    [[ "$modelName" == *"chopsticks"* ]] && \
        nohup $interpreter -m $trainer $expName $savedModel $freezeEncoder &>out/autoencoder/${modelName}_$stamp.out
    [[ "$modelName" == *"rope"* ]] && \
        nohup $interpreter -m $trainer $expName $savedModel $freezeEncoder &>out/autoencoder/${modelName}_$stamp.out
done