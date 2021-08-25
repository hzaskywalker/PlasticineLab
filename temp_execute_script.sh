export CUDA_VISIBLE_DEVICES=2

if [ ${#ENV_NAME} -eq 0 ]; then
	ENV_NAME="Chopsticks-v2"
fi
ENV_lower=${ENV_NAME,,}
echo "The env to execute is $ENV_NAME, the IOU and the rewards will be stored as td3_${ENV_lower}_XXX"

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_normal --pretrained_model="pretrain_model/cfm_v2.pth" --num_steps 50000 &> solve_0.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_normal --pretrained_model="pretrain_model/cfm_v2.pth" --num_steps 50000 &> solve_0.out
fi

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_linear --pretrained_model="pretrain_model/linear_v2.pth" --num_steps 50000 &> solve_1.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_linear --pretrained_model="pretrain_model/linear_v2.pth" --num_steps 50000 &> solve_1.out
fi

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_simple --pretrained_model="pretrain_model/simple_v2.pth" --num_steps 50000 &> solve_2.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_simple --pretrained_model="pretrain_model/simple_v2.pth" --num_steps 50000 &> solve_2.out
fi 

echo "python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_origin_loss --pretrained_model="pretrain_model/origin_loss_v2.pth" --num_steps 50000 &> solve_3.out"
if [ $1 != "--dry-run" ]; then
	python3 -m plb.algorithms.solve --env_name $ENV_NAME --algo td3 --srl --result_name td3_${ENV_lower}_origin_loss --pretrained_model="pretrain_model/origin_loss_v2.pth" --num_steps 50000 &> solve_3.out
fi
