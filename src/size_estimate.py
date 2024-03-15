import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                # raise NotImplementedError
                # TODO: fill-in (end)
                
                
            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                raise NotImplementedError
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (end)
                raise NotImplementedError
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (end)
                raise NotImplementedError
                # TODO: fill-in (end)
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


    # memory_usage = []

    # def hook_fn(m, input, output):
    #     if isinstance(output, (tuple, list)):
    #         for o in output:
    #             memory_usage.append(o.nelement() * o.element_size())
    #     else:
    #         memory_usage.append(output.nelement() * output.element_size())
        
    #     for i in input:
    #         if isinstance(i, torch.Tensor):
    #             memory_usage.append(i.nelement() * i.element_size())

    # hooks = []
    # for layer in model.modules():
    #     hooks.append(layer.register_forward_hook(hook_fn))
    
    # input_tensor = torch.rand(*input_shape).to(device)
    # with torch.no_grad():
    #     model(input_tensor)
    
    # for hook in hooks:
    #     hook.remove()

    # total_memory_usage = sum(memory_usage)
    # return total_memory_usage