import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data, batch_size: int = 256, device='cpu'):
    module.set_quant_state(True, True)
    calibration_device = next(module.parameters()).device if list(module.parameters()) else torch.device(device)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)

    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            input_batch = cali_data[i * batch_size:(i + 1) * batch_size].to(calibration_device)
            module(input_batch)
    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)
