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
    

# DATA_PATH = '/data/datasets/mimetics/'

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream: # and float(video_stream['duration']) > 1.1
            return True
        else:
            return False
    except:
        return False

def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=-1)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_files = ['/'.join(vid_paths[i].replace('\\', '/').split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files


mimetics50 = [ 'archery',     'playing_accordion',
 'bowling',                         'playing_basketball',
 'brushing_hair',                   'playing_bass_guitar',
 'brushing_teeth',                  'playing_guitar',
 'canoeing_or_kayaking',            'playing_piano',
 'catching_or_throwing_baseball',  'playing_saxophone',
 'catching_or_throwing_frisbee',    'playing_tennis',
 'clean_and_jerk',                  'playing_trumpet',
 'cleaning_windows',                'playing_violin',
 'climbing_a_rope',                 'playing_volleyball',
 'climbing_ladder',                'punching_person_(boxing)',
 'deadlifting',                     'reading_book',
 'dribbling_basketball',            'reading_newspaper',
 'drinking',                        'shooting_basketball',
 'driving_car',                    'shooting_goal_(soccer)',
 'dunking_basketball',             'skiing_(not_slalom_or_crosscountry)',
 'eating_cake',                     'skiing_slalom',
 'eating_ice_cream',                'skipping_rope',
 'flying_kite',                     'smoking',
 'golf_driving',                   'surfing_water',
 'hitting_baseball',                'sweeping_floor',
 'hurdling',                        'sword_fighting',
 'juggling_balls',                  'tying_tie',
 'juggling_soccer_ball',            'walking_the_dog',
 'opening_bottle',                  'writing',
]

mimetics10 = [ # https://arxiv.org/pdf/1910.02806.pdf
 'canoeing_or_kayaking', 'climbing_a_rope',
 'driving_car', 'golf_driving',
 'opening_bottle', 'playing_piano',
 'playing_volleyball', 'shooting_goal_(soccer)',
 'surfing_water',  'writing',
]


class Mimetics(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 split,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_labels=False,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 subset='mimetics50', # mimetics50, mimetics10
                 ):
        # if it throws error discard the problemetic videos by modifying filter videos codes.
        assert subset in ['mimetics50', 'mimetics10'], f"unknown {subset}"
        assert split in ['val', 'test', 'ood_test'], f"unknown {split}"
        split = 'val'
        
        ROOT = os.path.join(f"{DATA_PATH}", "videos")
        classes = sorted(eval(subset))
        
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'mimetics', "val.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        else:
            all_files = [fn for fn in glob.glob(os.path.join(f"{ROOT}", "*","*.mp4"))] # files related to mimetics50
            files = filter_videos(all_files) # load files that has both audios and videos
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(files))
            
        filenames = [fn for fn in files if fn.split('/')[-2] in classes]
        labels = [classes.index(fn.split('/')[-2]) for fn in filenames]
            
        super(Mimetics, self).__init__(
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

        self.name = 'Mimetics dataset'
        self.root = ROOT
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
