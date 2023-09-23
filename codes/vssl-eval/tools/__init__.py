from .logger import Logger, ProgressMeter, AverageMeter, accuracy, mean_ap_metric, synchronize_holder, all_evals, calculate_prec_recall_f1
from .utils import resume_model, save_checkpoint, get_parent_dir, set_deterministic, sanity_check
from .paths import my_paths, return_home
import torch
import torch.nn as nn
import os
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F 

########### common stuff

def set_grad(nets, requires_grad=False):
    for param in nets.parameters():
        param.requires_grad = requires_grad
            
########### finetune stuff

def warmup_cosine_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
    cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    return lr_schedule

def warmup_multistep_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, milestones, gamma, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    milestones = [iter_per_epoch * m for m in milestones]
    total_iter = iter_per_epoch * num_epochs
    multistep_lr_schedule = []
    lr = base_lr
    for i in range(warmup_iter, total_iter):
        if i in milestones:
            lr = lr*gamma
        multistep_lr_schedule.append(lr)
    multistep_lr_schedule = np.array(multistep_lr_schedule)
    lr_schedule = np.concatenate((warmup_lr_schedule, multistep_lr_schedule))

    return lr_schedule

def warmup_fixed_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    
    fixed_iters = (num_epochs - warmup_epochs) * iter_per_epoch
    fixed_lr_schedule = np.ones(fixed_iters) * base_lr
    lr_schedule = np.concatenate((warmup_lr_schedule, fixed_lr_schedule))

    return lr_schedule

def get_params_groups(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # print(n)

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
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
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

def save_checkpoint(args, classifier, optimizer, epoch, name='classifier'):
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, name + ".pth.tar")
    
    checkpoint = {'optimizer': optimizer.state_dict(), 
                'classifier': classifier.state_dict(), 
                'epoch': epoch + 1}

    torch.save(checkpoint, model_path)
    print(f"Classifier saved to {model_path}")

    
class Classifier(nn.Module):
    "classifier head"
    def __init__(self, num_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
        super(Classifier, self).__init__()
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
    
    
def LBR(in_dim, out_dim):
    """
    configuration for <fc + batchnorm + relu>
    """
    return nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        ])

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, nlayers=3):
        super().__init__()
        
        layers = nn.ModuleList()
        for k in range(nlayers):
            layers.extend(LBR(in_dim, hidden_dim))
            in_dim=hidden_dim
        
        layers.extend(
            nn.ModuleList([
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            ])
        )
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class BaseHead(nn.Module):
    def __init__(self, base, head,
                 ):
        super().__init__()
        
        self.base = base
        self.head = head

    def forward(self, x):
        
        x = self.base(x)
        x = self.head(x)
        return x
    

############## feature extraction

class Feature_Bank(object):
   
    def __init__(self, world_size, distributed, net, logger, print_freq=10, mode='vid', l2_norm=True):

        # mode = vid or aud
        self.mode = mode
        self.world_size = world_size
        self.distributed = distributed
        self.net = net
        self.logger = logger
        self.print_freq = print_freq
        self.l2_norm = l2_norm
        
    @torch.no_grad()
    def fill_memory_bank(self, data_loader):
            
        feature_bank = []
        feature_labels = []
        feature_indexs = []
        self.logger.add_line("Extracting features...")
        phase = 'test_dense' if data_loader.dataset.mode == 'video' else None
        
        for it, sample in enumerate(data_loader):
            if self.mode == 'vid':
                data = sample['frames'] 
            elif self.mode == 'aud':
                data = sample['audio']

            target = sample['label'].cuda(non_blocking=True)
            index = sample['index'].cuda(non_blocking=True)
            
            if phase == 'test_dense':
                batch_size, clips_per_sample = data.shape[0], data.shape[1]
                data = data.flatten(0, 1).contiguous()
                
            feature = self.net(data.cuda(non_blocking=True)).detach()
            if self.l2_norm:
                feature = F.normalize(feature, dim=1) # l2 normalize
            if feature.shape[0]==1:
                pass
            else:
                feature = torch.squeeze(feature)
            
            if phase == 'test_dense':
                feature = feature.view(batch_size, clips_per_sample, -1).contiguous()
                
            if self.distributed:
                # create blank tensor
                sub_feature_bank    = [torch.ones_like(feature) for _ in range(self.world_size)]
                sub_labels_bank     = [torch.ones_like(target) for _ in range(self.world_size)]
                sub_index_bank      = [torch.ones_like(index) for _ in range(self.world_size)]
                # gather from all processes
                dist.all_gather(sub_feature_bank, feature)
                dist.all_gather(sub_labels_bank, target)
                dist.all_gather(sub_index_bank, index)
                # concat them 
                sub_feature_bank = torch.cat(sub_feature_bank)
                sub_labels_bank = torch.cat(sub_labels_bank)
                sub_index_bank = torch.cat(sub_index_bank)
                # append to one bank in all processes
                feature_bank.append(sub_feature_bank.contiguous().cpu())
                feature_labels.append(sub_labels_bank.cpu())
                feature_indexs.append(sub_index_bank.cpu())
                
            else:
                
                feature_bank.append(feature.contiguous().cpu())
                feature_labels.append(target.cpu())
                feature_indexs.append(index.cpu())
            
            # print(feature.shape)
            if it%100==0:
                self.logger.add_line(f'{it} / {len(data_loader)}')
                    
        feature_bank    = torch.cat(feature_bank, dim=0)
        feature_labels  = torch.cat(feature_labels)
        feature_indexs  = torch.cat(feature_indexs)
        
        return feature_bank, feature_labels, feature_indexs
    
    
    
    
    