import os
import glob
import numpy as np
import re
import json
import torch  
try:
    from .video_db import VideoDataset
except:
    from video_db import VideoDataset

def get_dataset_original(root, name, 
                dataset_kwargs, subset,
                video_transform=None, 
                split='train'):
    
    """ with original splits
    """
    from datasets.loader.ucf import UCF
    from datasets.loader.hmdb import HMDB
    from datasets.loader.kinetics import Kinetics
        
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = dataset_kwargs[subset]['split'].format(fold=dataset_kwargs[subset]['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs[split]['clip_duration'],
                 video_fps=dataset_kwargs[split]['video_fps'],
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
                 subset = dataset_kwargs[subset]['split'].format(fold=dataset_kwargs[subset]['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs[split]['clip_duration'],
                 video_fps=dataset_kwargs[split]['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    elif name=='mitv2':
        raise NotImplementedError("Need to code this...")
    
    elif name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  video_clip_duration=dataset_kwargs[split]['clip_duration'],
                  video_fps=dataset_kwargs[split]['video_fps'],
                  video_transform=video_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    



    else:
        raise NotImplementedError(f'{name} is not available.')
        



def get_osar_val_dataset(root, name, subset, 
                dataset_kwargs, 
                video_transform=None, 
                ):
    
    """ with modified splits
    """
          
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = subset,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs['mode'],
                 clips_per_video=dataset_kwargs['clips_per_video'],
                 )

    elif name=='hmdb51':
        return HMDB(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = subset,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs['mode'],
                 clips_per_video=dataset_kwargs['clips_per_video'],
                 )
    

    
    else:
        raise NotImplementedError(f'{name} is not available.')
        


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'HMDB-51')
# ANNO_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'testTrainMulti_7030_splits')
# subset = 'train-split1'

class HMDB(VideoDataset):
    """ Test split for UCF101-OSAR and Kinetics400-OSAR"""
    def __init__(self, 
                  DATA_PATH,
                  ANNO_PATH,
                  subset,
                  video_clip_duration=1.,
                  video_fps=25.,
                  video_transform=None,
                  mode='clip',
                  clips_per_video=20,
                  ):
                
        self.name = 'HMDB-51'
        self.root = DATA_PATH
        self.subset = subset
        
        assert subset in ['test-k400', 'test-u101']
        if subset == 'test-k400':
            classes = ['Chew', 'Climb_Stairs', 'Draw_Sword', 'Fall_Floor', 'Fencing', 'Flic_Flac', 'Handstand', 'Hit', 'Jump', 'Kick', 'Pick', 'Pour', 'Run', 'Sit', 'Shoot_Gun', 'Smile', 'Stand', 'Sword_Exercise', 'Talk', 'Turn', 'Walk', 'Wave']
        elif subset == 'test-u101':
            classes = ['Brush_Hair', 'Cartwheel', 'Chew', 'Clap', 'Climb_Stairs', 'Drink', 'Eat', 'Fall_Floor', 'Flic_Flac', 'Hit', 'Hug', 'Kick', 'Kiss', 'Laugh', 'Pick', 'Pour', 'Push', 'Run', 'Shake_Hands', 'Shoot_Ball', 'Shoot_Bow', 'Shoot_Gun', 'Sit', 'Situp', 'Smile', 'Smoke', 'Somersault', 'Stand', 'Swing_Baseball', 'Sword', 'Sword_Exercise', 'Talk', 'Turn', 'Wave']
        else:
            raise NotImplementedError()
        
        # subset_id = {'train': '1', 'test': '2'}[subset]
        classes = sorted([k.lower() for k in classes])
        
        # sanity all classes are in all_classes
        all_classes = sorted(os.listdir(DATA_PATH))
        for k in classes:
            assert k in all_classes
            

        # in OSAR not using separate fold
        filenames = [ln.split('/')[-2:] for ln in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi"))\
                     if ln.split('/')[-2] in classes]
        filenames = [os.path.join(p[0], p[1]) for p in filenames]
        label_names = [fn.split('/')[0].replace('_', ' ')
                      for fn in filenames] 
        

        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        
        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = classes

        super(HMDB, self).__init__(
            return_video=True,
            video_clip_duration=video_clip_duration,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=False,
            labels=labels,
            label_names=label_names,
            # class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'UCF-101')
# ANNO_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'ucfTrainTestlist')
# subset = 'trainlist01'

class UCF(VideoDataset):
    """ Test split for Kinetics400-OSAR """
    def __init__(self, 
                 DATA_PATH, 
                 ANNO_PATH,
                 subset,
                 video_clip_duration=0.5,
                 video_fps=16.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=20,
                 ):
        
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset
        
        assert subset in ['test-k400']
        
        
        classes_fn = os.path.join(ANNO_PATH, 'classInd.txt')
        if subset == 'test-k400':
            classes = ['Apply_Lipstick', 'Balance_Beam', 'Billiards', 'Blow_Dry_Hair', 'Fencing', 'Field_Hockey_Penalty', 'Front_Crawl', 'Hammering', 'Handstand_Pushups', 'Handstand_Walking', 'Horse_Race', 'Ice_Dancing', 'Jumping_Jack', 'Military_Parade', 'Mixing', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Daf', 'Playing_Dhol', 'Playing_Sitar', 'Playing_Tabla', 'Pommel_Horse', 'Rafting', 'Rowing', 'Still_Rings', 'Sumo_Wrestling', 'Table_Tennis_Shot', 'Uneven_Bars', 'Wall_Pushups', 'YoYo']
        else:
            raise NotImplementedError()
                       
        
        classes = sorted([k.replace('_', '') for k in classes])
        # sanity all classes are in all_classes
        all_classes = [l.strip().split()[1] for l in open(classes_fn)]
        for k in classes:
            assert k in all_classes
                                        
        # in OSAR not using separate fold
        filenames = [ln.split('/')[-2:] for ln in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi"))\
                     if ln.split('/')[-2] in classes]
            
        filenames = [os.path.join(p[0], p[1]) for p in filenames]
        label_names = [' '.join(re.findall('[A-Z][^A-Z]*', fn.replace('\\', '/').split('/')[0])).lower()
                      for fn in filenames] 
        
        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        
        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes  = classes

        super(UCF, self).__init__(
            return_video=True,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=False,
            labels=labels,
            label_names=label_names,
            mode=mode,
            clips_per_video=clips_per_video,
        )




