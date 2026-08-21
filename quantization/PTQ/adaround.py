import os
import torch
from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.batch_norm_fold import fold_all_batch_norms
from aimet_torch.v2.adaround.adaround_weight import Adaround, AdaroundParameters



def pass_calibration_data(sim_model, args):
    test_set, device = args
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    sim_model.eval().to(device)
    samples = 1000
    batch_cntr = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            sim_model(images)

            batch_cntr += 1
            if batch_cntr * 64 >= samples:
                break

def adaround2(fp_model,test_set,args,label_key=None):

    # _ = fold_all_batch_norms(model, input_shapes=(1, 3, 32, 32))
    # sim = QuantizationSimModel(
    #     model=model,
    #     quant_scheme=QuantScheme.post_training_tf,
    #     default_output_bw=8,
    #     default_param_bw=8,
    #     dummy_input=torch.randn(1, 3, 32, 32).to(device),
    # )
    # sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=(data['test_set'], device))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    params = AdaroundParameters(
        data_loader=test_loader, num_batches=1, default_num_iterations=32
    )
    print("params--------------------------:",params.__dict__)
    if args['dataset_name'] == 'UTKFace' or args['dataset_name'] == 'celebA':
        dummy_input = torch.randn(1, 3, 224, 224).to(args['device'])
    elif args['dataset_name'] == 'ATnT':
        dummy_input = torch.randn(1, 1, 112, 92).to(args['device'])
    elif args['dataset_name'] == 'fmnist' or args['dataset_name']=='mnist':
        dummy_input = torch.randn(1, 1, 28, 28).to(args['device'])
    elif args['dataset_name'] == 'texas':
        dummy_input = torch.randn(1, 6169).to(args['device'])
    elif args['dataset_name'] == 'location':
        dummy_input = torch.randn(1, 168).to(args['device'])
    elif args['dataset_name'] == 'ATnT_binary_random':
        dummy_input = torch.randn(1, 1, 50, 60).to(args['device'])
    else:
        dummy_input = torch.randn(1, 3, 32, 32).to(args['device'])

    ada_model = Adaround.apply_adaround(
        fp_model, dummy_input, params, path='adaround_output',
        filename_prefix='adaround', default_param_bw=args['w_bit'],
        default_quant_scheme=QuantScheme.post_training_tf_enhanced
    )

    sim = QuantizationSimModel(model=ada_model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_output_bw=args['a_bit'],
                           default_param_bw=args['w_bit'])

    sim.set_and_freeze_param_encodings(encoding_path=os.path.join("adaround_output", 'adaround.encodings'))
    sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=(test_set, args['device']))

    return sim.model

