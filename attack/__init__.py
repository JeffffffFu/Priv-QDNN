def get_attack_method(attack_method,net_name,dataset_name):
    if attack_method == 'MIA':
        from attack.MIA import MIA, MIA_for_dp

        if net_name in ('mlp_dp', 'mlp_5layer_dp', 'resnet18_dp', 'CNN_UTKFace_dp'):
            return MIA_for_dp
        else:
            return MIA
    elif attack_method=='RMIA':
        from attack.RMIA import RMIA

        return RMIA
    elif attack_method=='WhiteBoxMIA':
        from attack.WhiteBox_MIA import WhiteBox_MIA

        return WhiteBox_MIA
    elif attack_method=='WhiteBoxMIA_Partial':
        from attack.WhiteBox_MIA import WhiteBox_MIA_Partial

        return WhiteBox_MIA_Partial
    else:
        raise NotImplementedError(f" {attack_method} not implemented!")
