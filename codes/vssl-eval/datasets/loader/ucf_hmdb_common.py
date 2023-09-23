import os
import glob
import numpy as np
import re
import json
import torch  
from datasets.loader.backend.video_db import VideoDataset


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'HMDB-51')
# ANNO_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'testTrainMulti_7030_splits')
# subset = 'train-split1'

class HMDB(VideoDataset):
    def __init__(self, 
                  DATA_PATH,
                  ANNO_PATH,
                  # split,
                  subset,
                  video_clip_duration=1.,
                  video_fps=25.,
                  video_transform=None,
                  mode='clip',
                  clips_per_video=20,
                  ):
                
        self.name = 'HMDB-UCF-COMMON'
        self.root = DATA_PATH
        self.subset = subset
        subset, split = subset.split('-')
        subset_id = {'train': '1', 'test': '2'}[subset]
        
        classes = {
            'catch': 0,
             'climb': 1,#
             'dive': 2,
             'dribble': 3,
             'fencing': 4,
             'golf': 5,
             'handstand': 6,
             'jump': 7,
             'kick_ball': 8,#
             'pullup': 9,
             'punch': 10,
             'pushup': 11,
             'ride_bike': 12,#
             'ride_horse': 13,#
             'throw': 14,
             'walk': 15,
             'shoot_bow': 16,
             }

        # sanity all classes are in all_classes
        all_classes = sorted(os.listdir(DATA_PATH))
        for k in classes:
            assert k in all_classes
            

        # # in OSAR not using separate fold
        # filenames = [ln.split('/')[-2:] for ln in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi"))\
        #              if ln.split('/')[-2] in classes]
        # filenames = [os.path.join(p[0], p[1]) for p in filenames]
        
        filenames, labels = [], []
        for cls in classes:
            for ln in open(os.path.join(f'{ANNO_PATH}', f'{cls}_test_{split}.txt')): 
                fn, ss = ln.strip().split()
                if ss == subset_id:
                    filenames += [os.path.join(f"{cls}", f"{fn}")]
                    labels += [classes[cls]]
        
        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = list(classes.keys())

        super(HMDB, self).__init__(
            return_video=True,
            video_clip_duration=video_clip_duration,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            # return_tokens=False,
            labels=labels,
            # label_names=label_names,
            mode=mode,
            clips_per_video=clips_per_video,
        )


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'UCF-101')
# ANNO_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'ucfTrainTestlist')
# subset = 'trainlist01'

class UCF(VideoDataset):
    def __init__(self, 
                 DATA_PATH, 
                 ANNO_PATH,
                 # split,
                 subset,
                 video_clip_duration=0.5,
                 video_fps=16.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=20,
                 ):
        
        self.name = 'UCF-HMDB-COMMON'
        self.root = DATA_PATH
        self.subset = subset
        
        classes = {
            'FrisbeeCatch': 0,
             'RockClimbingIndoor': 1,
             'Diving': 2,
             'BasketballDunk': 3,
             'Basketball': 3,
             'Fencing': 4,
             'GolfSwing': 5,
             'HandstandPushups': 6,
             'HandstandWalking': 6,
             'LongJump': 7,
             'JumpingJack': 7,
             'SoccerPenalty': 8,
             'PullUps': 9,
             'Punch': 10,
             'PushUps': 11,
             'Biking': 12,
             'HorseRiding': 13,
             'ThrowDiscus': 14,
             'WalkingWithDog': 15,
             'Archery': 16,
             }
        
        classes_fn = os.path.join(ANNO_PATH, 'classInd.txt')
        all_classes = [l.strip().split()[1] for l in open(classes_fn)]
        
        # classes = [k.replace('_', '') for k in classes]
        # sanity all classes are in all_classes
        for k in classes:
            assert k in all_classes
                                        
        filenames = [ln.strip().split()[0].split('/') for ln in open(os.path.join(f'{ANNO_PATH}', f'{subset}.txt'))
                     if ln.split('/')[-2] in classes]
        filenames = [os.path.join(p[0], p[1]) for p in filenames]
        labels = [fn.split('/')[0] for fn in filenames]  
        labels = [classes[ll] for ll in labels]
                
        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes  = list(classes.keys())

        super(UCF, self).__init__(
            return_video=True,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            # return_tokens=False,
            labels=labels,
            # label_names=label_names,
            mode=mode,
            clips_per_video=clips_per_video,
        )
