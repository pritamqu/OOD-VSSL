import torch
import torch.nn as nn
import math
import torch.nn.functional as F 
try:
    from .modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit, \
        MultiViewWrapper, DINOHead, projector_dict, projection_MLP, prediction_MLP, \
            predictor_dict, dino_projector_dict
except:
    from modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit, \
        MultiViewWrapper, DINOHead, projector_dict, projection_MLP, prediction_MLP, \
            predictor_dict, dino_projector_dict
    
    
class VideoSimSiam(nn.Module):
    def __init__(self,
                 frame_size = (3, 224, 224),
                 num_frames = 32,
                 vid_patch_spatial = (16, 16),
                 vid_patch_temporal = 4,
                 encoder_cfg=None, 
                 projector_cfg=None,
                 predictor_cfg=None,
                 apply_cls_token=True,
                 masking_fn=None,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()


        self.encoder_dim = encoder_dict[encoder_cfg]['embed_dim']
        self.apply_cls_token = apply_cls_token
        
        self.encoder = vid_vit(
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
            pred_input_dim = projector_dict[projector_cfg]['out_dim']
        elif projector_cfg in dino_projector_dict:
            self.projector = DINOHead(in_dim=self.encoder_dim,
                    **dino_projector_dict[projector_cfg])
            pred_input_dim = dino_projector_dict[projector_cfg]['out_dim']
        else:
            raise ValueError('projector_cfg is not defined either in projector_dict or dino_projector_dict')
        
        self.predictor = prediction_MLP(in_dim=pred_input_dim, 
                                        **predictor_dict[predictor_cfg])
        
        self.frame_size = self.encoder.frame_size
        self.num_frames = self.encoder.num_frames
        self.patch_temporal = self.encoder.patch_temporal
        self.patch_spatial = self.encoder.patch_spatial
        self.patch_dim = self.encoder.patch_dim
        self.num_cuboids = self.encoder.num_cuboids # number of smaller cuboids
        # self.initialize_weights()

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
    
    def forward_encoder(self, x, mask_ratio):
        # x = self.encoder(x) # calling ViT_Backbone
        x = self.prepare_token(x, mask_ratio)
        x = self.encoder.encoder(x) # calling ViT_Backbone
        x = self.encoder.encoder_norm(x)
        if self.apply_cls_token:
            x = x[:, 1:, :].mean(dim=1) # mean without cls_token if present
        else:
            x = x.mean(dim=1)
        x = self.projector(x)
        return x

    def forward_loss(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        
    def forward(self, x1, x2, mask_ratio, **kwargs):
        if mask_ratio==0:
            mask_ratio=None
        z1, z2 = self.forward_encoder(x1, mask_ratio), self.forward_encoder(x2, mask_ratio)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = self.forward_loss(p1, z2) / 2 + self.forward_loss(p2, z1) / 2
        
        return {'loss': loss}
    
    def forward_features(self, x, mode='frames', **kwargs):
        x = self.encoder(x) # calling VideoViT
        return x

    def save_state_dicts(self, model_path):
        # """ custom function to save backbone for future use.
        # """
        torch.save(self.encoder.state_dict(), model_path+'_vid_backbone.pth.tar')

    def clip_gradients(self, clip):
        # adopted from DINO
        norms = []
        for name, p in self.encoder.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)       
                    
        return norms


if __name__=='__main__':
        
    frame_size = (3, 112, 112)
    num_frames = 8
    vid_patch_spatial = (16, 16)
    vid_patch_temporal = 4

  
    
    model = VideoSimSiam(frame_size = frame_size,
                    num_frames = num_frames,
                    vid_patch_spatial = vid_patch_spatial,
                    vid_patch_temporal = vid_patch_temporal,
                    encoder_cfg='tiny_encoder', 
                    projector_cfg='2048-gelu-3-256-8192-lastnorm',
                    predictor_cfg='8192-1-8192',
                    apply_cls_token=False,
                    )
    
    frames = torch.randn(2, 3, num_frames, frame_size[-1], frame_size[-1])
    
    op = model(frames, frames, 0.4
            )
    
