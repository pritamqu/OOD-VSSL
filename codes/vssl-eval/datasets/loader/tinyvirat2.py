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
    

# DATA_PATH = '/mnt/PS6T/datasets/Video/TinyVirat/TinyVIRAT_V2'
# splt = 'train'

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
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


def one_hot_encode(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i in range(len(labels)):
        ll = labels[i]
        for t in ll:
            y[i, t] = 1.0
    return y

def update_labels_based_on_subset(sub_list, full_list):
    new_sublist = []
    for sl in sub_list:
        if sl in full_list:
            new_sublist.append(sl)
    return new_sublist
        

class TinyViratv2(VideoDataset):
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
                 subset='mitv2', # TODO: take a subset of classes
                 ):
        
        """ average duration of 3 seconds """
        assert split in ['train', 'val', 'test', 'ood_test'], f"current split: {split}"
        if split == 'test' or split == 'ood_test': 
            split = 'val'

        ROOT = os.path.join(f"{DATA_PATH}", 'videos' , f"{split}")
        if subset == 'mitv2':
            class_dict = {'activity_carrying': 0,
             'Closing': 1,
             'Entering': 2,
             'Exiting': 3,
             'Loading': 4,
             'Opening': 5,
             'Pull': 6,
             'Push': 7,
             'Riding': 8,
             'activity_running': 9,
             'activity_standing': 10,
             'Talking': 11,
             'specialized_talking_phone': 12,
             'activity_walking': 13}
            classes = list(class_dict.keys()) # this list order are based on mitv2, won't match with the original order mentioned in json file
            num_classes = len(classes)
        else:
            class_dict = json.load(open(DATA_PATH+'/class_map.json', 'r'))
            classes = list(class_dict.keys())
            num_classes = len(classes)
        
        
        if split == 'train':
            annotations = json.load(open(DATA_PATH+'/tiny_train_v2.json', 'r'))
        elif split == 'val':
            annotations = json.load(open(DATA_PATH+'/tiny_val_v2.json', 'r'))
        else:
            raise NotImplementedError()
            # annotations = json.load(open(DATA_PATH+'/tiny_test_v2_public.json', 'r'))
            
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'tinyvirat', f"{split}.txt")
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                filtered=json.loads(f.read())
        else:
            # filenames = filter_videos(files) 
            all_files = [fn for fn in glob.glob(os.path.join(f"{ROOT}", "*","*.mp4"))]
            filtered = filter_videos(all_files)
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(filtered))   
            print(f"discarded {len(all_files)-len(filtered)} files")
            
        files = []
        labels = []
        for annotation in annotations:
            if annotation['path'] in filtered: # flterout corrupted files
                if subset=='mitv2':
                    # updating multi-label assignments based on available classes in the subset
                    tmp= update_labels_based_on_subset(annotation['label'], classes)
                    if len(tmp)>=1: 
                        files.append(annotation['path']) 
                        _label = [class_dict[ll] for ll in tmp]
                        labels.append(_label)
                        
                else: # original
                    # files.append(ROOT + '/' +annotation['path'])
                    files.append(annotation['path'])
                    _label = [class_dict[ll] for ll in annotation['label']]
                    labels.append(_label)
        
        # it's multi-label
        # filenames = filter_videos(files) 
        labels = one_hot_encode(labels, num_classes)
        filenames = files

        super(TinyViratv2, self).__init__(
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

        self.name = 'TinyViratv2 dataset'
        self.root = ROOT
        self.split = split
        self.subset = subset
        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
