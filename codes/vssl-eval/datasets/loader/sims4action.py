import os
import glob
import numpy as np
import random
import ffmpeg
import json
import pandas as pd
from joblib import Parallel, delayed
try:
    from datasets.loader.backend.video_db import VideoDataset
except:
    from backend.video_db import VideoDataset
    

# DATA_PATH = '/mnt/PS6T/datasets/Video/Sims4Action'
# splt = 'train'

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if video_stream and float(video_stream['duration']) > 1.1:
            return True
        else:
            return False
    except:
        return False

def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=-1)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_files = ['/'.join(vid_paths[i].replace('\\', '/').split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files

class Sims4Action(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 split,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 # return_audio=False,
                 # audio_clip_duration=1.,
                 # audio_fps=None,
                 # audio_fps_out=100,
                 # audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 # max_offsync_augm=0,
                 mode='clip',
                 # submode=None,
                 clips_per_video=1,
                 subset='mitv2', 
                 ):
        
        """ frame durations are 3 seconds """


        ROOT = os.path.join(f"{DATA_PATH}", 'Sims4ActionVideosProcessed')
        if subset == 'mitv2':
            class_dict = {'cook': 0,
             'drink': 1,
             'eat': 2,
             'read_book': 3,
             'use_phone': 4,
             'walk': 5}
            classes = [k.replace('_', '').capitalize() for k in class_dict] # old style
        else:
            classes = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
        assert split in ['train', 'val', 'test', 'ood_test'] # train is filtered
        if split in ['val', 'test', 'ood_test']:
            split='val'
        
        annotation = pd.read_csv(os.path.join(DATA_PATH,'Sims4ActionVideosProcessed','SimsSplitsChunks.csv'), delimiter=';')
        
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'sims4action', "filtered.txt")
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                filtered=json.loads(f.read())
        else:
            # filenames = filter_videos(files) 
            all_files = [fn for fn in glob.glob(os.path.join(f"{ROOT}", "*","*.avi"))]
            filtered = filter_videos(all_files)
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(filtered))   
        
        files, labels = [], []
        for file in annotation.values:
            if (file[1] + '/'+ file[0][:-4] + '_' + str(file[-5]) + '.avi') in filtered and file[8] == split and file[1] in classes: # for subset setup
                files.append(file[1] + '/'+ file[0][:-4] + '_' + str(file[-5]) + '.avi' )
                # files.append(ROOT + '/' + file[1] + '/'+ file[0][:-4] + '_' + str(file[-5]) + '.avi' )
                labels.append(classes.index(file[1]))
                
                
        # filenames = filter_videos(files) 
        filenames = files
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)
        self.subset = subset
        self.name = f'Sims4Action-{subset}'
        self.root = ROOT
        self.split = split
        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
  
        super(Sims4Action, self).__init__(
            return_video=return_video,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video,
        )

        







