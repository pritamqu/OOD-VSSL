from typing import Any
import random
import torch
import numpy as np


def get_vid_aug(name='standard', crop_size=224, num_frames=8, mode='train', aug_kwargs=None):

    from .video_augmentations import StrongTransforms

    if name == 'strong':
        augmentation = StrongTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
                    
    else:
        raise NotImplementedError


    
    return augmentation



