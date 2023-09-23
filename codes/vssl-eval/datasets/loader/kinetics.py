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
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream and video_stream and float(video_stream['duration']) > 1.1 and float(audio_stream['duration']) > 1.1:
            return True
        else:
            return False
    except:
        return False

def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=-1)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_files = ['/'.join(vid_paths[i].replace('\\', '/').split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files

class Kinetics(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_labels=False,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 ):

        ROOT = os.path.join(f"{DATA_PATH}", f"{subset}")
        classes = sorted(os.listdir(ROOT))
        
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{subset}.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        else:
            all_files = [fn for fn in glob.glob(os.path.join(f"{DATA_PATH}",f"{subset}", "*","*.avi"))]
            files = filter_videos(all_files) # load files that has both audios and videos
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(files))   
        
        filenames = files
        labels = [classes.index(fn.split('/')[-2]) for fn in filenames]
  
        super(Kinetics, self).__init__(
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

        self.name = 'Kinetics dataset'
        self.root = ROOT
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])


class KineticsSubset(VideoDataset):

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
        
        assert split in ['train', 'val', 'test']
        if split == 'test':
            split = 'val'
        
        assert subset in ['mimetics10', 'mimetics50', 'kinetics400', 'drone', 'actor_shift', 'game_play']
        if subset in ['mimetics10', 'mimetics50', 'kinetics400', 'drone', 'game_play']: # created from k400
            ROOT = os.path.join(f"{DATA_PATH}", split)
        elif subset in ['actor_shift', 'kinetics700',]: # created from k700
            ROOT = os.path.join(os.path.dirname(f"{DATA_PATH}"), 'kinetics700', split)
        
        

        mimetics10 = [ # https://arxiv.org/pdf/1910.02806.pdf
         'canoeing_or_kayaking', 'climbing_a_rope',
         'driving_car', 'golf_driving',
         'opening_bottle', 'playing_piano',
         'playing_volleyball', 'shooting_goal_(soccer)',
         'surfing_water',  'writing',
        ]
        mimetics50 = [ # https://arxiv.org/pdf/1912.07249.pdf
        'archery',     'playing_accordion',
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
                
        actor_shift = { # actorshift: https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Audio-Adaptive_Activity_Recognition_Across_Video_Domains_CVPR_2022_paper.pdf
                       'sleeping': 0,
                        'watching tv': 1,
                        'eating': 2,
                        'drinking': 3,
                        'swimming': 4,
                        'running': 5,
                        'opening door': 6,
                        }
        
        
        # all_classes =  sorted(os.listdir(ROOT))
        if subset == 'mimetics10':
            classes = sorted(mimetics10)
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{split}.txt")
        elif subset == 'mimetics50':
            classes = sorted(mimetics50)
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{split}.txt")
        elif subset == 'kinetics400':
            classes = sorted(os.listdir(ROOT))
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{split}.txt")
        elif subset == 'actor_shift':
            classes = list(actor_shift.keys())
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics700', f"actor_shift_{split}.txt")
        elif subset == 'kinetics700':
            classes = sorted(os.listdir(ROOT))
            CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics700', f"{split}.txt")
        else:
            raise NotImplementedError()
              
        
        # CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{split}.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        else:
            raise FileNotFoundError("prepare cache file separately")
            
        if subset=='actor_shift':
            filenames = [fn[0] for fn in files] # 276 missing files
            labels = [actor_shift[fn[1]] for fn in files]
        else:
            filenames = [fn for fn in files if fn.split('/')[-2] in classes]
            labels = [classes.index(fn.split('/')[-2]) for fn in filenames]
  
        super(KineticsSubset, self).__init__(
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

        self.name = f'Kinetics-{subset}'
        self.root = ROOT
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])

