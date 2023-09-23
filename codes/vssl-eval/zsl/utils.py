import torch.nn as nn

def LBR(in_dim, out_dim):
    """
    configuration for <fc + batchnorm + relu>
    """
    return nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        ])

    
def LR(in_dim, out_dim):
    """
    configuration for <fc + relu>
    """
    return nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        ])

class MLP(nn.Module):
    # following their setup: https://proceedings.neurips.cc/paper/2021/file/d6539d3b57159babf6a72e106beb45bd-Paper.pdf
    def __init__(self, in_dim, hidden_dim=1024, out_dim=768, nlayers=1, use_bn=False):
        super().__init__()
        
        self.out_dim=out_dim
        layers = nn.ModuleList()
        for k in range(nlayers):
            if use_bn:
                layers.extend(LBR(in_dim, hidden_dim))
            else:
                layers.extend(LR(in_dim, hidden_dim))
                
            in_dim=hidden_dim
        
        layers.extend(
            nn.ModuleList([
                nn.Linear(hidden_dim, out_dim),
            ])
        )
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.normalize(x, p=2, dim=-1)    
        return x
    
class VidText(nn.Module):
    "vid-text wrapper"
    def __init__(self, video_encoder, text_encoder, 
                  classifier='default',
                  vid_dropout=0.,
                  hidden_dim=1024,
                  num_classes=1,
                  ):
        super(VidText, self).__init__()
        
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        
        video_feature_dim = self.video_encoder.embed_dim
        text_feature_dim = self.text_encoder.out_dim
        
        if vid_dropout>0:
            self.vid_dropout = nn.Dropout(p=vid_dropout)
        else:
            self.vid_dropout = None
            
        self.v_to_joint = nn.Linear(video_feature_dim, hidden_dim)
        self.t_to_joint = nn.Linear(text_feature_dim, hidden_dim)
        
        if classifier == 'default':
            self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(True),
                    nn.Dropout(p=0.1),
                    nn.Linear(hidden_dim//2, num_classes)
                )
        else:
            self.classifier = classifier
                
    def forward(self, frames, text_embed, return_feats=False, **kwargs):
        # frames
        # text_embed -> class embed for all the classes; 361, 300
        
        
        vid_feat = self.video_encoder(frames, **kwargs) # output of vit
        # vid_feat normalized inside;
        if self.vid_dropout is not None: # not sure if required
            vid_feat = self.vid_dropout(vid_feat)
        
        # output of text refine module
        text_feat = self.text_encoder(text_embed) 
        # text_feat normalized inside;
        
        t_shape = text_feat.shape[0]
        v_shape = vid_feat.shape[0]
        # common space
        v_feats = nn.functional.normalize(self.v_to_joint(vid_feat)).unsqueeze(1).repeat(1, t_shape, 1) # b, c, n
        t_feats = nn.functional.normalize(self.t_to_joint(text_feat)).unsqueeze(0).repeat(v_shape, 1, 1) # b, c, n
        
        # v_feats = nn.functional.normalize(self.v_to_joint(vid_feat))
        # t_feats = nn.functional.normalize(self.t_to_joint(text_feat))
        # combined_feats = v_feats@t_feats.t()

        combined_feats = v_feats*t_feats
        
        out = self.classifier(combined_feats).squeeze(-1) 
        
        if return_feats:
            return out, vid_feat, text_feat
        return out
    
        
    
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

