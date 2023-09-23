import os
import glob
import numpy as np
import random
import ffmpeg
import json
from joblib import Parallel, delayed
try:
    from datasets.loader.backend.video_db import VideoDataset
except:
    from backend.video_db import VideoDataset
    

# DATA_PATH = '/data/datasets/kinetics/'

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if video_stream and float(video_stream['duration']) > 1.1 :
            return True
        else:
            return False
    except:
        return False

def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=-1)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_files = ['/'.join(vid_paths[i].replace('\\', '/').split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files


tiny2moments = {
      "Opening": 'opening', 
      "Interacts": None,
      "Pull": 'pulling',
      "activity_carrying": 'carrying',
      "Entering": 'entering', 
      "vehicle_moving": None,
      "Exiting": 'exiting',
      "Loading": 'loading',
      "Talking": 'talking',
      "activity_running": 'running',
      "vehicle_turning_left": None,
      "vehicle_stopping": None,
      "Riding": 'riding',
      "Closing": 'closing',
      "activity_walking": 'walking',
      "Push": 'pushing',
      "specialized_using_tool": None,
      "vehicle_starting": None,
      "specialized_miscellaneous": None,
      "activity_standing": 'standing',
      "Transport_HeavyCarry": None,
      "activity_gesturing": None,
      "vehicle_turning_right": None,
      "specialized_talking_phone": 'telephoning',
      "specialized_texting_phone": None,
      "Misc": None,
}


sims2moments = {
    'cook': 'cooking',
    'drink': 'drinking',
    'eat': 'eating', 
    'get_up_sit_down': None, 
    'read_book': 'reading', 
    'use_computer': None, 
    'use_phone': 'telephoning',
    'use_tablet': None, 
    'walk': 'walking',
    'watch_tv': None,
    }


class MiTv2(VideoDataset):

    def __init__(self,
                 DATA_PATH,
                 subset,
                 split,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_labels=True,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 ):
        
        """ frame durations are 3 seconds """
        
        assert split in ['train', 'val', 'test']
        if split == 'test' or split=='val':
            split = 'val'
            DATA_PATH = os.path.join(DATA_PATH, 'validation_processed')
        else:
            DATA_PATH = os.path.join(DATA_PATH, 'training_processed')
            
        assert subset in ['sims4action', 'tinyvirat'], f"current subset: {subset}"

        # all_classes =  sorted(os.listdir(ROOT))
        if subset == 'sims4action':
            classes = [sims2moments[k] for k in sims2moments if sims2moments[k] is not None]
            classes = sorted(classes)
            # class_dict = {k: classes.index(k) for k in classes}
            class_dict = {'cooking': 0,
             'drinking': 1,
             'eating': 2,
             'reading': 3,
             'telephoning': 4,
             'walking': 5}
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'mitv2', f"{subset}_{split}.txt")
        
        elif subset == 'tinyvirat':
            classes = [tiny2moments[k] for k in tiny2moments if tiny2moments[k] is not None]
            classes = sorted(classes)
            # class_dict = {k: classes.index(k) for k in classes}
            class_dict = {'carrying': 0,
             'closing': 1,
             'entering': 2,
             'exiting': 3,
             'loading': 4,
             'opening': 5,
             'pulling': 6,
             'pushing': 7,
             'riding': 8,
             'running': 9,
             'standing': 10,
             'talking': 11,
             'telephoning': 12,
             'walking': 13}
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'mitv2', f"{subset}_{split}.txt")
        
        else:
            raise NotImplementedError()
              
        
        # CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{split}.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        # else:
        #     raise FileNotFoundError("prepare cache file separately")
        else:
            all_files = [fn for fn in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi")) if fn.split('/')[-2] in classes]
            files = filter_videos(all_files) # load files which are not corrupted
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(files))   
                
        # filenames = [fn for fn in files if fn.split('/')[-2] in classes]
        filenames = files
        labels = [class_dict[fn.split('/')[-2]] for fn in filenames]
        
  
        super(MiTv2, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
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

        self.name = f'MiTv2-{subset}'
        self.root = DATA_PATH
        self.subset = subset
        self.split = split

        self.classes = list(class_dict.keys())
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
