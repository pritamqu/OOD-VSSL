import torch.nn as nn
import torch
from collections import OrderedDict

class OSClassifier(nn.Module):
    "classifier head"
    def __init__(self, num_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
        super(OSClassifier, self).__init__()
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        self.use_dropout = use_dropout
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._initialize_weights(self.classifier)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
                
    def forward(self, x):
        x = x.squeeze()
        # x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = nn.functional.normalize(x, p=2, dim=-1)        
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.classifier(x)
   

class VidOpenSet(nn.Module):
    "vid openset wrapper"
    def __init__(self, 
                 video_encoder, 
                 debias_head, 
                 classifier,
                 loss, 
                 ):
        super(VidOpenSet, self).__init__()
        
        # evidence loss loss is for open set
        # debias head is remove static bias
        
        self.video_encoder = video_encoder
        self.debias_head = debias_head
        self.classifier = classifier
        self.loss = loss
        
    def forward(self):
        # dummy 
        return 
                
    def forward_train(self, frames, target, 
                      
                      **kwargs):
        # frames
        losses = {}        
        vid_feat = self.video_encoder(frames, **kwargs) # output of vit

        if self.debias_head is not None:
            
            with torch.no_grad(): # to save computation
                frames_shuffled = frames[:, :, torch.randperm(frames.size()[2]), ::]
                vid_feat_shuffled = self.video_encoder(frames_shuffled, **kwargs)
                
                B, C, T, H, W = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3], frames.shape[4]
                frames_static = frames.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, 1, H, W)
                frames_static = frames_static.repeat(1, 1, 4, 1, 1) # our patch is 4x16, so repeat 4 times; this is an easy fix;  
                vid_feat_static = self.video_encoder(frames_static, **kwargs)
                vid_feat_static = vid_feat_static.view(B, T, -1).mean(1)
                # we can apply stop grad
            
            loss_debias = self.debias_head(vid_feat, vid_feat_shuffled.detach(), vid_feat_static.detach(),
                                       target=target.squeeze(), **kwargs)
            losses.update(loss_debias)
        
        logits = self.classifier(vid_feat)
        # target = target.unsqueeze(-1)
        loss_cls = self.loss(logits, target, **kwargs)
        losses.update(loss_cls)
        
        # loss
        loss, log_vars = self._parse_losses(losses)
        
        return loss, logits, log_vars
    
    def forward_test(self, frames, target, **kwargs):
        # frames
        vid_feat = self.video_encoder(frames, **kwargs) # output of vit
        # vid_feat normalized inside;
        
        logits = self.classifier(vid_feat)
        
        
        return logits
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            # if dist.is_available() and dist.is_initialized():
            #     loss_value = loss_value.data.clone()
            #     dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    
def get_params_groups(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    # vit related stuff
    video_params = ('video_encoder')
    num_layers = len(model.video_encoder.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    _tmp= no_weight_decay_list
    no_weight_decay_list = model.video_encoder.no_weight_decay()
    no_weight_decay_list = ['video_encoder.' + k for k in no_weight_decay_list]
    no_weight_decay_list.extend(_tmp)
    
    for n, p in model.named_parameters():
        
        if not p.requires_grad:
            continue
        # print(n)

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            # print('no decay', n)
            g_decay = "no_decay"
            this_decay = 0.
        else:
            # print('decay', n)
            g_decay = "decay"
            this_decay = weight_decay
        
        if n.startswith(video_params):
            layer_id = get_layer_id_for_vit(n, num_layers)
            group_name = "layer_%d_%s" % (layer_id, g_decay)
        else: # text encoder, classifier etc
            group_name = 'other_params_%s' %g_decay
            
        if group_name not in param_group_names:
            if group_name.startswith('other_params'):
                this_scale = 1. # no decay on the newly added params
            else:
                this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                # "name": 'video',
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                # "name": 'video',
            }
            
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())
    # return param_groups


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['video_encoder.cls_token', 'video_encoder.pos_embed']:
        return 0
    elif name.startswith('video_encoder.patch_embed'):
        return 0
    elif name.startswith('video_encoder.blocks'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers

