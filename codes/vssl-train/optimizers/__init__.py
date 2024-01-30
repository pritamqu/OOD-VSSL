import torch
from torch._six import inf
from .lr_scheduler import cosine_scheduler
from .larc import LARC
from timm.optim.optim_factory import add_weight_decay


def get_optimizer_nc(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):

    """ for simsiam, byol
    """

    # params
    predictor_prefix = ('module.predictor', 'predictor')
    def _get_params_groups(model):
        base_not_regularized = []
        base_regularized = []
        predictor_not_regularized = []
        predictor_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(predictor_prefix):
                # predictor.append(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    predictor_not_regularized.append(param)
                else:
                    predictor_regularized.append(param)
            else:
                # base.append(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    base_not_regularized.append(param)
                else:
                    base_regularized.append(param)
    
        return [
            {'name': 'base_0', 'params': base_not_regularized,},# 'weight_decay': 0.},
            {'name': 'base_wd', 'params': base_regularized, },
            {'name': 'predictor_0', 'params': predictor_not_regularized,},# 'weight_decay': 0.},
            {'name': 'predictor_wd', 'params': predictor_regularized, },
                ]
            
    # optimizer
    if name == 'adamw': # vits use adamw
        # add a separate weight decay scheduler
        if weight_decay is None:
            parameters = _get_params_groups(model)
        else:
            parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    
    elif name == 'adam': # resnets use adam
        parameters = _get_params_groups(model)
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    
    else:
        raise NotImplementedError

    return optimizer

def get_optimizer_w_pred(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):

    """ for simsiam, byol
    """

    # params
    predictor_prefix = ('module.predictor', 'predictor')
    def _get_params_groups(model):
        base_not_regularized = []
        base_regularized = []
        # predictor_not_regularized = []
        # predictor_regularized = []
        predictor = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(predictor_prefix):
                predictor.append(param)
                # if name.endswith(".bias") or len(param.shape) == 1:
                #     predictor_not_regularized.append(param)
                # else:
                #     predictor_regularized.append(param)
            else:
                # base.append(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    base_not_regularized.append(param)
                else:
                    base_regularized.append(param)
    
        return [
            {'name': 'base_0', 'params': base_not_regularized, 'weight_decay': 0.},
            {'name': 'base_wd', 'params': base_regularized, },
            # {'name': 'predictor_0', 'params': predictor_not_regularized, 'weight_decay': 0.},
            # {'name': 'predictor_wd', 'params': predictor_regularized, },
            {'name': 'predictor', 'params': predictor, 'weight_decay': 0.},
                ]
            
    # optimizer
    if name == 'adamw': # vits use adamw
        # add a separate weight decay scheduler
        if weight_decay is None:
            parameters = _get_params_groups(model)
        else:
            parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    
    # elif name == 'adam': # resnets use adam
    #     parameters = _get_params_groups(model)
    #     optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    
    else:
        raise NotImplementedError

    return optimizer


def get_optimizer_mae(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):
    """ for MAE, SimCLR, simple ViT setups
    """

    def _get_params_groups(model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [
            {'name': 'not_regularized', 'params': not_regularized, 'weight_decay': 0.}, 
            {'name': 'regularized', 'params': regularized}
                ]
    # optimizer
    if name == 'adamw':
        # following https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
        if weight_decay is None: # this is to add a different weight decay schedule
            parameters = _get_params_groups(model)
        else:
            parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    else:
        raise NotImplementedError

    return optimizer

# def get_optimizer_lr(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):

#     # optimizer
#     if name == 'adamw':
#         # following https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
#         if weight_decay is None: # this is to add a different weight decay schedule
#             parameters = get_params_groups_lr(model)
#         else:
#             parameters = add_weight_decay(model, weight_decay)
#         optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
#     else:
#         raise NotImplementedError

#     return optimizer



# def get_params_groups_lr(model):
    
#     vid_regularized = []
#     vid_not_regularized = []
    
#     aud_regularized = []
#     aud_not_regularized = []
    
#     vid_prefix = ('module.vid_')
#     aud_prefix = ('module.aud_')
    
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue

#         if name.startswith(vid_prefix):            
#             # we do not regularize biases nor Norm parameters
#             if name.endswith(".bias") or len(param.shape) == 1:
#                 vid_not_regularized.append(param)
#             else:
#                 vid_regularized.append(param)
                
#         elif name.startswith(aud_prefix):
#             # we do not regularize biases nor Norm parameters
#             if name.endswith(".bias") or len(param.shape) == 1:
#                 aud_not_regularized.append(param)
#             else:
#                 aud_regularized.append(param)

#         else:
#             # if something left off could be error
#             raise ValueError(f'{name} is not handled')
                
                
#     return [
#         {'name': 'video', 'params': vid_not_regularized, 'weight_decay': 0.}, 
#         {'name': 'video', 'params': vid_regularized},
#         {'name': 'audio', 'params': aud_not_regularized, 'weight_decay': 0.}, 
#         {'name': 'audio', 'params': aud_regularized},
#             ]

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
