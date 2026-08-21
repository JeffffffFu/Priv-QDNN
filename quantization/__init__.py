from quantization.PTQ.PYT_PTQ.PYT_PTQ import Asymmetric_quant
from quantization.PTQ.ada.adaround import adaround
from quantization.PTQ.adaround import adaround2
from quantization.PTQ.pd_quant.pd_quant import pd_quant
from quantization.PTQ.qdrop.qdrop import qdrop


def _adaround_router(model, test_set, args, label_key=None):
    """Route AdaRound: ada PTQ for mlp_5layer or bitwidth < 4; AIMET for >= 4 elsewhere."""
    a_bit = int(args.get('a_bit', 8))
    w_bit = int(args.get('w_bit', 8))
    if args.get('net_name') == 'mlp_5layer' or a_bit < 4 or w_bit < 4:
        return adaround(model, test_set, args, label_key=label_key)
    return adaround2(model, test_set, args, label_key=label_key)

from quantization.QAT.PYT_QAT import PYT_QAT
from quantization.QAT.bsq.bsq_adv_aia import bsq_adv_aia
from quantization.QAT.bsq.bsq import bsq
from quantization.QAT.bsq.bsq_adv_mia import bsq_adv_mia
from quantization.QAT.bsq.bsq_dp import bsq_dp
from quantization.QAT.bsq.bsq_vib_minv import bsq_vib
from quantization.QAT.hawq.hawq import hawq
from quantization.QAT.hawq.hawq_adv_aia import hawq_adv_aia
from quantization.QAT.hawq.hawq_adv_mia import hawq_adv_mia
from quantization.QAT.hawq.hawq_dp import hawq_dp
from quantization.QAT.hawq.hawq_vib_minv import hawq_vib
from quantization.QAT.pact.pact import pact
from quantization.QAT.pact.pact_adv_aia import pact_adv_aia
from quantization.QAT.pact.pact_adv_mia import pact_adv_mia
from quantization.QAT.pact.pact_dp import pact_dp
from quantization.QAT.pact.pact_vib_minv import pact_vib


def get_PTQ_quantization(quantizaiton_method,net_name='None'):

    if quantizaiton_method == 'PYT_PTQ':
        return Asymmetric_quant
    elif quantizaiton_method=='adaround':
        return _adaround_router
    elif quantizaiton_method=='pd_quant':
        return pd_quant
    elif quantizaiton_method=='qdrop':
        return qdrop
    elif quantizaiton_method=='pact':

        if net_name in ('resnet18_dp', 'CNN_UTKFace_dp', 'mlp_dp', 'mlp_5layer_dp'):
            return pact_dp
        elif net_name=='CNN_UTKFace_adversarial':
            return pact_adv_aia
        elif net_name=='resnet18_adversarial':
            return pact_adv_mia
        elif net_name == 'mlp_vib':
            return pact_vib
        else:
            return pact
    elif quantizaiton_method=='PYT_QAT':
        return PYT_QAT
    elif quantizaiton_method=='hawq':
        if net_name in ('resnet18_dp', 'CNN_UTKFace_dp', 'mlp_dp', 'mlp_5layer_dp', 'CNN_MNIST_dp'):
            return hawq_dp
        elif net_name=='CNN_UTKFace_adversarial':
            return hawq_adv_aia
        elif net_name in ('resnet18_adversarial', 'mlp_5layer_adversarial'):

            return hawq_adv_mia

        elif net_name == 'CNN_MNIST_adversarial':
            return hawq_vib
        else:
            return hawq
    elif quantizaiton_method=='bsq' :
        if net_name in ('resnet18_dp', 'CNN_UTKFace_dp', 'mlp_dp', 'mlp_5layer_dp', 'CNN_MNIST_dp'):
            return bsq_dp
        elif net_name=='CNN_UTKFace_adversarial':
            return bsq_adv_aia
        elif net_name=='resnet18_adversarial':
            return bsq_adv_mia
        elif net_name == 'CNN_MNIST_adversarial':
            return bsq_vib
        else:
            return bsq
    else:
        raise NotImplementedError(f" {quantizaiton_method} not implemented!")
