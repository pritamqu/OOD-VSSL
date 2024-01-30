import torch
import torch.nn as nn
import math
import torch.nn.functional as F 
import torch.distributed as dist

try:
    from .modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit, \
        MultiViewWrapper, DINOHead, projector_dict, projection_MLP, prediction_MLP, \
            predictor_dict, dino_projector_dict
except:
    from modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit, \
        MultiViewWrapper, DINOHead, projector_dict, projection_MLP, prediction_MLP, \
            predictor_dict, dino_projector_dict

class DINOLoss(nn.Module):
    # originally copied from https://github.com/facebookresearch/dino/blob/main/main_dino.py#L363
    # and modified based on our setup
    def __init__(self, out_dim, 
                 center_momentum=0.9):
        super().__init__()

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum

    def forward(self, student_output, teacher_output, 
                student_temp, teacher_temp, 
                num_student_views, num_teacher_views, 
                
                ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / student_temp
        student_out = student_out.chunk(num_student_views)

        # teacher centering and sharpening
        # temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(num_teacher_views)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        if dist.is_initialized(): # default case
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)             # local debug 

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
class BaseHead(nn.Module):
    def __init__(self, base, head,
                 ):
        super().__init__()
        
        self.base = base
        self.head = head

    def forward(self, x):
        
        x = self.base.encoder(x) # calling ViT_Backbone
        x = self.base.encoder_norm(x)
        if self.base.apply_cls_token:
            x = x[:, 1:, :].mean(dim=1) # mean without cls_token if present
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        return x
    
class VideoDINO(nn.Module):
    def __init__(self,
                 frame_size = (3, 224, 224),
                 num_frames = 32,
                 vid_patch_spatial = (16, 16),
                 vid_patch_temporal = 4,
                 encoder_cfg=None, 
                 projector_cfg=None,
                 center_momentum=0.9,
                 apply_cls_token=True,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()


        self.encoder_dim = encoder_dict[encoder_cfg]['embed_dim']
        self.apply_cls_token = apply_cls_token
        
        self.encoder = vid_vit( # main video-vit backbone
                     frame_size = frame_size,
                     num_frames = num_frames,
                     patch_spatial = vid_patch_spatial,
                     patch_temporal = vid_patch_temporal,
                     encoder_cfg=encoder_cfg,
                     apply_cls_token=apply_cls_token,
                     norm_layer=norm_layer,
                     )
        
        if projector_cfg in projector_dict:
            self.projector = projection_MLP(in_dim=self.encoder_dim,
                    **projector_dict[projector_cfg])
            target_projector = projection_MLP(in_dim=self.encoder_dim,
                    **projector_dict[projector_cfg])
            
            proj_out_dim = projector_dict[projector_cfg]['out_dim']
            
        elif projector_cfg in dino_projector_dict:
            self.projector = DINOHead(in_dim=self.encoder_dim,
                    **dino_projector_dict[projector_cfg])
            
            target_projector = DINOHead(in_dim=self.encoder_dim,
                    **dino_projector_dict[projector_cfg])
            
            proj_out_dim = dino_projector_dict[projector_cfg]['out_dim']
        else:
            raise ValueError('projector_cfg is not defined either in projector_dict or dino_projector_dict')
        
        self.online_encoder = BaseHead(
            self.encoder,
            self.projector
        )
        
        #### copy of online encoder
        self.target_encoder = BaseHead(
            vid_vit( 
                    frame_size = frame_size,
                    num_frames = num_frames,
                    patch_spatial = vid_patch_spatial,
                    patch_temporal = vid_patch_temporal,
                    encoder_cfg=encoder_cfg,
                    apply_cls_token=apply_cls_token,
                    norm_layer=norm_layer,
                    ),
            target_projector
        )
        
        self.register_buffer("center", torch.zeros(1, proj_out_dim))
        self.center_momentum = center_momentum
        
        self.frame_size = self.encoder.frame_size
        self.num_frames = self.encoder.num_frames
        self.patch_temporal = self.encoder.patch_temporal
        self.patch_spatial = self.encoder.patch_spatial
        self.patch_dim = self.encoder.patch_dim
        self.num_cuboids = self.encoder.num_cuboids # number of smaller cuboids
        # self.initialize_weights()
        self.init_target_online_same_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                
    def init_target_online_same_weights(self):
        self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=False)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, m):
        # ema
        for param_v, param_k in zip(
                                self.online_encoder.parameters(), 
                                self.target_encoder.parameters()):

            param_k.data.mul_(m).add_((1 - m) * param_v.detach().data)
            
    def masking_fn(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = round(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def prepare_token(self, x, mask_ratio=None):
               
        B, nc, t, w, h = x.shape
        target, x = self.encoder.cuboid_embed(x)
        pos_embed = self.encoder.interpolate_pos_encoding(x, t, w, h)
        
        # add pos embed w/o cls token
        if self.apply_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed
            
        # apply mask
        if mask_ratio is not None:
            x, mask, ids_restore = self.masking_fn(x, mask_ratio)
        
        # append cls token
        if self.apply_cls_token:
            cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        return x
   
    
    def forward_loss(self, teacher_output, student_output, teacher_temp, student_temp):
        # this setup is comparable to other methods; different from DINO, not using multi-view etc.
        
        t = teacher_output
        s = student_output / student_temp
        
        t = F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
        loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1)
        
        self.update_center(teacher_output)
        
        return loss.mean()
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        if dist.is_initialized(): # default case
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)             # local debug 

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
    
    def forward(self, x1, x2, mask_ratio, m, 
                  student_temp, teacher_temp,
                  **kwargs):
        if mask_ratio==0:
            mask_ratio=None
        
        z1 = self.online_encoder(self.prepare_token(x1, mask_ratio))
        z2 = self.online_encoder(self.prepare_token(x2, mask_ratio))
        
        with torch.no_grad():
            self.update_target(m)
            # not applying masking, it does not make sense applying masking in no-grad
            zz1 = self.target_encoder(self.prepare_token(x1))
            zz2 = self.target_encoder(self.prepare_token(x2))
        
        loss = self.forward_loss(teacher_output=zz2, student_output=z1, teacher_temp=teacher_temp, student_temp=student_temp) / 2 + \
            self.forward_loss(teacher_output=zz1, student_output=z2, teacher_temp=teacher_temp, student_temp=student_temp) / 2
                    
        return {'loss': loss}
    
    def forward_features(self, x, mode='frames', **kwargs):
        x = self.encoder(x) # calling VideoViT
        return x

    def save_state_dicts(self, model_path):
        """ custom function to save backbone for future use.
        """
        torch.save(self.encoder.state_dict(), model_path+'_vid_backbone.pth.tar')

    def clip_gradients(self, clip):
        # adopted from DINO
        norms = []
        for name, p in self.online_encoder.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)       
                    
        return norms

    def cancel_gradients_last_layer(self, epoch, freeze_last_layer):
        # on projector # adopted from DINO
        if epoch >= freeze_last_layer:
            return
        for n, p in self.online_encoder.head.named_parameters():
            if "last_layer" in n:
                p.grad = None
       

if __name__=='__main__':
        
    frame_size = (3, 112, 112)
    num_frames = 8
    vid_patch_spatial = (16, 16)
    vid_patch_temporal = 4

  
    
    model = VideoDINO(frame_size = frame_size,
                    num_frames = num_frames,
                    vid_patch_spatial = vid_patch_spatial,
                    vid_patch_temporal = vid_patch_temporal,
                    encoder_cfg='tiny_encoder', 
                    projector_cfg='2048-3-2048',
                    predictor_cfg='2048-1-2048',
                    apply_cls_token=False,
                    )
    
    frames_s = torch.randn(2, 3, num_frames, frame_size[-1], frame_size[-1])
    frames_t = torch.randn(2, 3, num_frames, frame_size[-1], frame_size[-1])
    
    op = model(frames_s, frames_t, mask_ratio=0.5, m=0.996
            )
    