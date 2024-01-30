import torch
import torch.nn as nn
import math
try: 
    from .modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit
except:
    from modules import PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict, vid_vit

class VideoMAE(nn.Module):
    def __init__(self,
                 frame_size = (3, 224, 224),
                 num_frames = 32,
                 vid_patch_spatial = (16, 16),
                 vid_patch_temporal = 4,
                 encoder_cfg=None, 
                 decoder_cfg=None,
                 apply_cls_token=True,
                 masking_fn=None,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=True):
        super().__init__()


        self.encoder_dim = encoder_dict[encoder_cfg]['embed_dim']
        self.decoder_dim = decoder_dict[decoder_cfg]['embed_dim']
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
        self.decoder = ViT_Backbone(**decoder_dict[decoder_cfg], mlp_ratio=4)


        self.frame_size = self.encoder.frame_size
        self.num_frames = self.encoder.num_frames
        self.patch_temporal = self.encoder.patch_temporal
        self.patch_spatial = self.encoder.patch_spatial
        self.patch_dim = self.encoder.patch_dim
        self.num_cuboids = self.encoder.num_cuboids # number of smaller cuboids

        if self.apply_cls_token:
            self.dec_pos_embed = PosEmbedding(self.num_cuboids+1, self.decoder_dim).pos_embed
        else:
            self.dec_pos_embed = PosEmbedding(self.num_cuboids, self.decoder_dim).pos_embed


        self.mid_proj = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True) if self.encoder_dim != self.decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.out_proj = nn.Linear(self.decoder_dim, self.patch_dim) if self.decoder_dim != self.patch_dim else nn.Identity()

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
                
    def masking_fn(self, x, mask_ratio): # flop cal
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        if torch.is_tensor(mask_ratio):
            mask_ratio=mask_ratio.cpu().numpy()[0]
        
        N, L, D = x.shape  # batch, length, dim
        
        N = N.cpu().numpy() if torch.is_tensor(N) else N
        L = L.cpu().numpy() if torch.is_tensor(L) else L
        D = D.cpu().numpy() if torch.is_tensor(D) else D
        
        
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
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def prepare_token(self, x, mask_ratio, **kwargs):
               
        B, nc, t, w, h = x.shape
        target, x = self.encoder.cuboid_embed(x)
        pos_embed = self.encoder.interpolate_pos_encoding(x, t, w, h)
        
        # add pos embed w/o cls token
        if self.apply_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed
        
        # apply mask
        x, mask, ids_restore = self.masking_fn(x, mask_ratio, **kwargs)
            
        # append cls token
        if self.apply_cls_token:
            cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        return x, target, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio, **kwargs):
        x, target, mask, ids_restore = self.prepare_token(x, mask_ratio, **kwargs)
        x = self.encoder.encoder(x) # calling ViT_Backbone
        x = self.encoder.encoder_norm(x)
        return x, target, mask, ids_restore

    def forward_decoder(self, x, t, w, h, ids_restore):
        # embed tokens
        x = self.mid_proj(x)
        # append mask tokens to sequence
        if self.apply_cls_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # tackling cls token # appending mask token with encoder op
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # appending mask token with encoder op

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.apply_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_

        x = x + self.dec_pos_embed

        x = self.decoder(x) # apply Transformer blocks
        x = self.out_proj(x) # predictor projection
        if self.apply_cls_token:
            x = x[:, 1:, :] # remove cls token
        
        return x

    def forward_loss(self, target, pred, mask):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, frames, mask_ratio, return_latent=False, **kwargs):
        _, _, t, w, h = frames.shape
        latent, target, mask, ids_restore = self.forward_encoder(frames, mask_ratio, **kwargs)
        pred = self.forward_decoder(latent, t, w, h, ids_restore)
        loss = self.forward_loss(target, pred, mask)
        if return_latent:
            return latent, loss
        return {'loss': loss}
    
    def forward_features(self, x, mode='frames', **kwargs):
        x = self.encoder(x) # calling VideoViT
        return x

    def save_state_dicts(self, model_path):
        # """ custom function to save backbone for future use.
        # """
        torch.save(self.encoder.state_dict(), model_path+'_vid_backbone.pth.tar')



if __name__=='__main__':
        
    frame_size = (3, 112, 112)
    num_frames = 8
    vid_patch_spatial = (16, 16)
    vid_patch_temporal = 4

  
    
    model = VideoMAE(frame_size = frame_size,
                    num_frames = num_frames,
                    vid_patch_spatial = vid_patch_spatial,
                    vid_patch_temporal = vid_patch_temporal,
                    encoder_cfg='tiny_encoder', 
                    decoder_cfg='tiny_decoder',
                    apply_cls_token=False,
                    norm_pix_loss=True,
                    )
    
    frames = torch.randn(2, 3, num_frames, frame_size[-1], frame_size[-1])
    
    op = model(frames,
               mask_ratio=0.90, 
            )
    