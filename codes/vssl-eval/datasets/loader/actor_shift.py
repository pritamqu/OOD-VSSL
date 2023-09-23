import os
import pandas as pd
import numpy as np
from datasets.loader.backend.video_db import VideoDataset


# from tools.paths import my_paths
# DATA_PATH = my_paths('local', 'actor_shift')[-1]
# subset = 'train'

class ActorShift(VideoDataset):
    def __init__(self, 
                 DATA_PATH, 
                 split,
                 subset='',
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode='clip',
                 clips_per_video=20,
                 ):

        assert return_audio is False
        self.name = 'ActorShift'
        self.root = os.path.join(DATA_PATH, 'videos_processed')
        self.subset = subset
        class_dict = {'sleeping': 0,
                        'watching tv': 1,
                        'eating': 2,
                        'drinking': 3,
                        'swimming': 4,
                        'running': 5,
                        'opening door': 6,
                        }
        if split in ['test', 'ood_test', 'val']:
            split = 'val'
        anno_path = os.path.join(f'{DATA_PATH}', 'annotations', f'{split}.csv')
        entries = pd.read_csv(anno_path, skiprows=0, header=None).values
        
        filenames = [os.path.join(self.root, k[0].split('/')[1])[:-4]+'.avi' for k in entries]
        labels = [class_dict[k[1]] for k in entries]
   
        self.num_classes = len(class_dict)
        self.num_videos = len(filenames)
        self.classes = list(class_dict.keys())

        super(ActorShift, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=return_labels,
            return_index=return_index,
            labels=labels,
            mode=mode,
            clips_per_video=clips_per_video,
        )

