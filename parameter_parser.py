import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--dataset_name', '--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cinic10', 'texas', 'location'],
                        dest='dataset_name')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Choose the  device")
    parser.add_argument('--training', type=str, default='None', choices=[ 'None', 'target', 'shadow','both'])


    ######################### quantization related parameters ################################

    parser.add_argument('--quantization', type=str, default='Original', choices=[ 'Original', 'PYT_PTQ', 'QAT','pact','adaround','pd_quant','qdrop','PYT_QAT','hawq','bsq'])
    parser.add_argument('--a_bit', type=int, default=32)
    parser.add_argument('--w_bit', type=int, default=32)

    parser.add_argument('--random', type=int, default=41)
    ######################### target model related parameters ################################
    parser.add_argument('--net_name', type=str, default='resnet18',
                        choices=['DT', 'mlp', 'mlp_5layer', 'mlp_5layer_dp', 'mlp_5layer_drop010', 'mlp_5layer_drop015', 'mlp_5layer_drop020', 'mlp_5layer_adversarial', 'LR', 'RF', 'LRTorch', 'MLPTorch', 'simple_cnn', 'resnet18', 'mobilenetv2', 'resnet18_hawq', 'resnet18_ATnT', 'resnet18_bsq','resnet20','vgg','resnet18_dp','resnet50', 'densenet','CNN_UTKFace','simple_cnn_dropout','LeNet','resnet18_drop010','resnet18_drop015','resnet18_drop020','resnet18_drop010_full','resnet18_drop015_full','resnet18_drop020_full','CNN_UTKFace_dp','mlp_dp','CNN_UTKFace_dropout','mlp_dropout','CNN_UTKFace_adversarial','mlp_vib','resnet18_adversarial','CNN_MNIST','CNN_MNIST_dp','CNN_MNIST_adversarial','CNN_MNIST_memorization','CNN_MNIST_HSIC'])

    parser.add_argument('--attack_method', type=str, default='None',
                        choices=['MIA','RMIA','IDLG','WhiteBoxMIA','WhiteBoxMIA_Partial','None'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: .1)", )
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)

    ######################### attack related parameters ################################
    parser.add_argument('--trials', type=int, default=5,
                        help="number of trials")
    parser.add_argument('--observations', type=int, default=1,
                        help="number of observations")
    parser.add_argument(
        '--compleaknr_attacks',
        type=str,
        default='all',
        choices=['all', 'confidence'],
        help="MIA attack-only mode: run all six black-box variants, or only the confidence threshold attack (faster).",
    )


    args = vars(parser.parse_args())

    return args