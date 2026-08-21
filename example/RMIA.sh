
#RMIA don't need --training, it will train when it attack, They are simultaneous
python main.py --attack_method RMIA --dataset cifar100 --net_name resnet18_hawq --device cuda:3 --observations 64

python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18  --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --observations 64  --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 1

#adaround only stpport 4-32 bit
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2  --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --observations 64  --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 1

## qdrop
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 1

## pact
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 1

#
## hawq
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --observations 64 --device cuda:3 --trials 1

#bsq
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10  --observations 64 --device cuda:3 --trials 3
python main.py  --attack_method RMIA --dataset cifar100 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6  --observations 64 --device cuda:3 --trials 3

python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --observations 64  --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2  --observations 64 --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --observations 64  --device cuda:3 --trials 1
python main.py  --attack_method RMIA --dataset cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --observations 64  --device cuda:3 --trials 1
