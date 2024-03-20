import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates the number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                input_features = module.in_features
                output_features = module.out_features
                bias_ops = output_features if module.bias is not None else 0
                flops[module] = batch_size * (2 * input_features * output_features + bias_ops)

            if isinstance(module, nn.Conv2d):
                # in_channels = module.in_channels
                # out_channels = module.out_channels
                # kernel_h, kernel_w = module.kernel_size
                # output_height, output_width = output.shape[2:]
                # groups = module.groups
                # bias_ops = out_channels if module.bias is not None else 0
                # flops[module] = batch_size * ((2 * kernel_h * kernel_w * in_channels // groups) * output_height * output_width * out_channels + bias_ops)
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_h, kernel_w = module.kernel_size
                output_height, output_width = output.shape[2:]
                groups = module.groups
                # Correcting the calculation here
                conv_ops = batch_size * (kernel_h * kernel_w * in_channels // groups) * output_height * output_width * out_channels
                bias_ops = output_height * output_width * out_channels if module.bias is not None else 0
                flops[module] = (2 * conv_ops) + bias_ops  


            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                num_elements = input[0].numel()  
                affine_ops = 2 * num_elements if module.affine else 0  
                flops[module] = 2 * num_elements + affine_ops  

            if isinstance(module, nn.Softmax):
                num_elements = input[0].numel()  
                flops[module] = 3 * num_elements  

            total[name] = flops
        
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total

def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    # raise NotImplementedError
    # TODO: fill-in (end)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    
    # TODO: fill-in (start)
    # raise NotImplementedError
    # TODO: fill-in (end)
    # Register hook to capture memory usage of intermediate tensors
    
    input_tensor = torch.rand(input_shape).to(device)
    input_mem = np.prod(input_shape) * 4
    model = model.to(device)
    output_mem = np.prod(model(input_tensor).size()) * 4

    return input_mem + output_mem
