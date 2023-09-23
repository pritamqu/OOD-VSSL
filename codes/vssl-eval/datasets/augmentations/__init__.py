def get_vid_aug(name='standard', crop_size=224, num_frames=8, mode='train', aug_kwargs=None):

    from .video_augmentations import StrongTransforms, StrongTransforms3Crop, \
        RandVisTransforms, RandVisTransforms3Crop

    if name == 'strong':
        augmentation = StrongTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'strong3crop':
        augmentation = StrongTransforms3Crop(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'randaug':
        augmentation = RandVisTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'randaug3crop':
        augmentation = RandVisTransforms3Crop(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
    
    else:
        raise NotImplementedError

    
    return augmentation


