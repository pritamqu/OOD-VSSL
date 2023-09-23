import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

# supportive func of vid-text training

# https://github.com/showlab/EgoVLP/blob/main/model/loss.py
class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
    
    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
    
class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )
    
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


class VidText(nn.Module):
    "vid-text wrapper"
    # ref: https://github.com/showlab/EgoVLP/blob/main/model/model.py#L85
    def __init__(self, video_encoder, text_encoder, 
                 projection_dim=256,
                  ):
        super(VidText, self).__init__()
        
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder

        video_feature_dim = self.video_encoder.embed_dim
        text_feature_dim = self.text_encoder.config.hidden_size
        
            
        self.v_to_joint = nn.Linear(video_feature_dim, projection_dim)
        self.t_to_joint = nn.Sequential(nn.ReLU(),
                                        nn.Linear(text_feature_dim, projection_dim),
                                     )

    
    
    
    def forward(self, frames, text, **kwargs):
        # frames
        vid_feat = self.video_encoder(frames, **kwargs) # output of vit
        
        # distilbert
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0, :]

        # common space
        # v_feats = nn.functional.normalize(self.v_to_joint(vid_feat))
        # t_feats = nn.functional.normalize(self.t_to_joint(text_feat))
        
        # normalizing later under sim_matrix
        v_feats = self.v_to_joint(vid_feat)
        t_feats = self.t_to_joint(text_feat)

        return t_feats, v_feats
        
    def forward_text(self, text):
       
        # distilbert
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0, :]

        # common space
        # t_feats = nn.functional.normalize(self.t_to_joint(text_feat))
        
        # normalizing later under sim_matrix
        t_feats = self.t_to_joint(text_feat)

        return t_feats
    
    def forward_video(self, frames, **kwargs):
        # frames
        vid_feat = self.video_encoder(frames, **kwargs) # output of vit
        
        # common space
        # v_feats = nn.functional.normalize(self.v_to_joint(vid_feat))
        
        # normalizing later under sim_matrix
        v_feats = self.v_to_joint(vid_feat)

        return v_feats
    