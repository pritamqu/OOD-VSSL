import os
from datasets.loader.kinetics import Kinetics, KineticsSubset
from datasets.loader.ucf import UCF
from datasets.loader.hmdb import HMDB
from datasets.loader.charadesego_text import CharadesEgoText
from datasets.loader.mimetics import Mimetics
from datasets.loader.tinyvirat2 import TinyViratv2 
from datasets.loader.sims4action import Sims4Action
from datasets.loader.actor_shift import ActorShift
from datasets.loader.moments_in_time import MiTv2
from datasets.loader.ucf_hmdb_common import HMDB as HMDBCOMMON, UCF as UCFCOMMON

import random
import torch


def get_dataset(root, dataset_kwargs, video_transform=None, audio_transform=None, split='train'):
    name = dataset_kwargs['name']
          
    ## action recognition
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)

    elif name=='hmdb51':
        return HMDB(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
            
    elif name=='charadesego_text':
        return CharadesEgoText(
            root = root,
                 # anno = os.path.join(root, 'CharadesEgo', 'video-action'),
                 subset = dataset_kwargs[split]['subset'], # autoset split based on subset
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 # return_labels=True,
                 # return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)  
    
    elif name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  # return_video=dataset_kwargs['return_video'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    elif name=='mimetics':
        return Mimetics(
            DATA_PATH = os.path.join(root),
                  split = split,
                  subset = dataset_kwargs[split]['subset'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )
    
    elif name=='kinetics_subset':
        return KineticsSubset(
            DATA_PATH = os.path.join(root),
                  split = split,
                  subset = dataset_kwargs[split]['subset'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )
    
    elif name=='tinyvirat':
        return TinyViratv2(
            DATA_PATH = os.path.join(root),
                  split = split,
                  subset = dataset_kwargs[split]['subset'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )

    elif name=='sims4action':
        return Sims4Action(
            DATA_PATH = os.path.join(root),
                  split = split,
                  subset = dataset_kwargs[split]['subset'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )
    
    elif name=='actor_shift':
        return ActorShift(
            DATA_PATH = os.path.join(root),
                  split = split,
                  # subset = dataset_kwargs[split]['split'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )

    elif name=='mitv2':
        return MiTv2(
            DATA_PATH = os.path.join(root),
                  split = split,
                  subset = dataset_kwargs[split]['subset'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],
                  )

    elif name=='ucf-hmdb':
        return UCFCOMMON(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
        
    elif name=='hmdb-ucf':
        return HMDBCOMMON(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    else:
        raise NotImplementedError(f'{name} is not available.')
        
class FetchSubset(torch.utils.data.Subset):

    def __init__(self, dataset, size=None):
        self.dataset = dataset
        if size is None:
            size = len(dataset.classes)
        self.indices = random.sample(range(len(dataset)), size)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dataset, name)
    
