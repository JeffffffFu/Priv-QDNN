from attack import get_attack_method
from parameter_parser import parameter_parser
from training import get_training_method_target, get_training_method_shadow


def main(args):

    if args['training']=='None':
        attack=get_attack_method(args['attack_method'],args['net_name'],args['dataset_name'])
        attack(args)

    else:

        if args['training']=='both':
            train_method = get_training_method_target(args)
            train_method(args)
            train_method = get_training_method_shadow(args)
            train_method(args)
        elif args['training']=='target':
            train_method = get_training_method_target(args)
            train_method(args)
        elif args['training']=='shadow':
            train_method = get_training_method_shadow(args)
            train_method(args)
        else:
            raise ValueError("this algorithm is not exist")


if __name__ == '__main__':
    args = parameter_parser()
    main(args)
