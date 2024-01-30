import os
from datasets.loader.kinetics import Kinetics
import random
import torch


def get_dataset(root, dataset_kwargs, video_transform=None, audio_transform=None, split='train'):
    name = dataset_kwargs['name']
              
    if name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  return_video=dataset_kwargs['return_video'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_audio=dataset_kwargs['return_audio'],
                  audio_clip_duration=dataset_kwargs['audio_clip_duration'] if dataset_kwargs['return_audio'] else dataset_kwargs['clip_duration'],
                  audio_fps=dataset_kwargs['audio_fps'] if dataset_kwargs['return_audio'] else None,
                  audio_fps_out= dataset_kwargs['audio_fps_out'] if dataset_kwargs['return_audio'] else None, # if 'audio_fps_out' in dataset_kwargs else dataset_kwargs['audio_fps'], # when extracting raw waveforms
                  audio_transform=audio_transform,
                  return_labels=dataset_kwargs['return_labels'] if 'return_labels' in dataset_kwargs else False,
                  return_index=dataset_kwargs['return_index'] if 'return_index' in dataset_kwargs else False,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)

    else:
        raise NotImplementedError
        
# used for a quick test in debug mode
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