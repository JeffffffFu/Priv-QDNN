from training.train import model_training_target, model_training_shadow, model_training_shadow_RMIA, \
    model_training_target_RMIA
from training.train_with_Minv_memorization import MInv_memorization
from training.train_with_adversarial_training import  model_training_with_adversarial
from training.train_with_dp import model_training_target_dp


def get_training_method_target(args):
    if args['attack_method'] == 'RMIA':
        return model_training_target_RMIA
    elif args['net_name'] in ('resnet18_dp', 'CNN_UTKFace_dp', 'CNN_MNIST_dp', 'mlp_5layer_dp'):
        return model_training_target_dp
    elif args['net_name'] in (
        'CNN_UTKFace_adversarial',
        'resnet18_adversarial',
        'CNN_MNIST_adversarial',
        'CNN_MNIST_HSIC',
        'mlp_5layer_adversarial',
    ):
        return model_training_with_adversarial
    elif args['net_name']=='CNN_MNIST_memorization':
        return MInv_memorization
    else:
        return model_training_target

def get_training_method_shadow(args):

    if args['attack_method'] == 'RMIA':
        return model_training_shadow_RMIA
    else:
        return model_training_shadow
