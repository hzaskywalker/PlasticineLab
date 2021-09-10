if [ ${#CUDA_VISIBLE_DEVICES} -eq 0 ]; then
	CUDA_VISIBLE_DEVICES=0
fi

if [ ${#ENV_NAME} -eq 0 ]; then
	ENV_NAME="Chopsticks-v1"
fi
ENV_lower=${ENV_NAME,,}
ENV_lower=${ENV_lower%-*}
echo "The env to execute is $ENV_NAME, the IOU and the rewards will be stored as td3_${ENV_lower}_XXX"

mkdir -p ious
mkdir -p rewards

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_normal --model_name=${ENV_lower}_cfm --num_steps 50000 &> out/td3/${ENV_lower}_0.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_normal --model_name="${ENV_lower}_cfm" --num_steps 50000 &> out/td3/${ENV_lower}_0.out
fi

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_linear --model_name=${ENV_lower}_linear --num_steps 50000 &> out/td3/${ENV_lower}_1.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_linear --model_name="${ENV_lower}_linear" --num_steps 50000 &> out/td3/${ENV_lower}_1.out
fi

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_simple --model_name=${ENV_lower}_smaller --num_steps 50000 &> out/td3/${ENV_lower}_2.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_simple --model_name="${ENV_lower}_smaller" --num_steps 50000 &> out/td3/${ENV_lower}_2.out
fi 

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_origin_loss --model_name=${ENV_lower}_origin_loss --num_steps 50000 &> out/td3/${ENV_lower}_3.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --exp_name td3_${ENV_lower}_origin_loss --model_name="${ENV_lower}_origin_loss" --num_steps 50000 &> out/td3/${ENV_lower}_3.out
fi
