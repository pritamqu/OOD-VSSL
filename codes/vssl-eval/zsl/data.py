import os
import glob
import numpy as np
import re
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import torch  
try:
    from .video_db import VideoDataset
except:
    from video_db import VideoDataset

def get_dataset(root, name, subset, 
                dataset_kwargs, 
                video_transform=None, 
                wv_model = None,
                classes2embedding=None,
                ):
          
    ## action recognition
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = subset.format(fold=dataset_kwargs['fold']),
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs['mode'],
                 clips_per_video=dataset_kwargs['clips_per_video'],
                 wv_model=wv_model,
                 classes2embedding=classes2embedding,
                 )

    elif name=='hmdb51':
        return HMDB(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = subset.format(fold=dataset_kwargs['fold']),
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 mode=dataset_kwargs['mode'],
                 clips_per_video=dataset_kwargs['clips_per_video'],
                 wv_model=wv_model,
                 classes2embedding=classes2embedding,
                 )
    
    elif name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = subset,
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  mode=dataset_kwargs['mode'],
                  clips_per_video=dataset_kwargs['clips_per_video'],
                  wv_model=wv_model,
                  classes2embedding=classes2embedding,
                  )

    elif name=='kinetics700':
        return Kinetics700(
            DATA_PATH = os.path.join(root),
                  subset = subset,
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  mode=dataset_kwargs['mode'],
                  clips_per_video=dataset_kwargs['clips_per_video'],
                  wv_model=wv_model,
                  classes2embedding=classes2embedding,
                  )
    
    elif name=='mimetics':
        return Mimetics(
            DATA_PATH = os.path.join(root),
                  subset = subset,
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  mode=dataset_kwargs['mode'],
                  clips_per_video=dataset_kwargs['clips_per_video'],
                  wv_model=wv_model,
                  classes2embedding=classes2embedding,
                  )
    
    elif name=='rareact':
        return RareAct(
            DATA_PATH = os.path.join(root),
                  subset = subset,
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  mode=dataset_kwargs['mode'],
                  clips_per_video=dataset_kwargs['clips_per_video'],
                  wv_model=wv_model,
                  classes2embedding=classes2embedding,
                  )
    
    else:
        raise NotImplementedError(f'{name} is not available.')
        


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'HMDB-51')
# ANNO_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'testTrainMulti_7030_splits')
# subset = 'train-split1'

class HMDB(VideoDataset):
    """ ZSL DB """
    def __init__(self, 
                  DATA_PATH,
                  ANNO_PATH,
                  subset,
                  video_clip_duration=1.,
                  video_fps=25.,
                  video_transform=None,
                  mode='clip',
                  clips_per_video=20,
                  wv_model=None,
                  classes2embedding=None,
                  ):
        
        # splits are based on: https://github.com/kini5gowda/TruZe/blob/main/ZSL.txt
        HMDB_ZSL={ # zero shot based on kinetics400
        'train':['Brush_Hair', 'Cartwheel', 'Catch', 'Clap', 'Climb', 'Dive', 'Dribble', 'Drink', 'Eat', 'Golf', 'Hug', 'Kick_Ball', 'Kiss', 'Laugh', 'Pullup', 'Punch', 'Push', 'Pushup', 'Ride_Bike', 'Ride_Horse', 'Shoot_Ball', 'Shake_Hands', 'Shoot_Bow', 'Situp', 'Somersault', 'Swing_Baseball', 'Smoke', 'Sword', 'Throw'],
        'test':['Chew', 'Climb_Stairs', 'Draw_Sword', 'Fall_Floor', 'Fencing', 'Flic_Flac', 'Handstand', 'Hit', 'Jump', 'Kick', 'Pick', 'Pour', 'Run', 'Sit', 'Shoot_Gun', 'Smile', 'Stand', 'Sword_Exercise', 'Talk', 'Turn', 'Walk', 'Wave'],
        }
        
        self.name = 'HMDB-51'
        self.root = DATA_PATH
        self.subset = subset
        # subset_id = {'train': '1', 'test': '2'}[subset]
        
        if subset == 'train':
            classes = sorted([k.lower() for k in HMDB_ZSL['train']])
        elif subset == 'test':
            classes = sorted([k.lower() for k in HMDB_ZSL['test']])
        elif subset == 'all':
            # https://arxiv.org/pdf/2003.01455.pdf; Table 2: 4.3. Evaluation protocol -> Evaluation Protocol 2:
            _tmp = HMDB_ZSL['test']
            _tmp.extend(HMDB_ZSL['train'])
            classes = sorted([k.lower() for k in _tmp])
        else:
            raise NotImplementedError()
            
            
        # sanity all classes are in all_classes
        all_classes = sorted(os.listdir(DATA_PATH))
        for k in classes:
            assert k in all_classes
            
        filenames = [ln.split('/')[-2:] for ln in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi"))\
                     if ln.split('/')[-2] in classes]
        filenames = [os.path.join(p[0], p[1]) for p in filenames]
        label_names = [fn.split('/')[0].replace('_', ' ')
                      for fn in filenames] 
        

        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('hmdb51', classes, wv_model)
        if isinstance(class_embeddings, list): # classes2embedding2 returns 2
            assert len(class_embeddings)==2
            label_names=class_embeddings[1] # words that fed to the word2vec
            class_embeddings=class_embeddings[0]
        
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
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'UCF-101')
# ANNO_PATH = os.path.join(my_paths('local', 'ucf101')[-1], 'ucfTrainTestlist')
# subset = 'trainlist01'

class UCF(VideoDataset):
    """ ZSL DB """
    def __init__(self, 
                 DATA_PATH, 
                 ANNO_PATH,
                 subset,
                 video_clip_duration=0.5,
                 video_fps=16.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=20,
                 wv_model=None,
                 classes2embedding=None,
                 ):
        
        # splits are based on: https://github.com/kini5gowda/TruZe/blob/main/ZSL.txt
        UCF_ZSL={ # zero shot based on kinetics400
        'train':['Apply_Eye_Makeup', 'Archery', 'Baby_Crawling', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Blowing_Candles', 'Cutting_In_Kitchen', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Diving', 'Drumming', 'Floor_Gymnastics', 'Frisbee_Catch', 'Golf_Swing', 'Haircut', 'Hammer_Throw', 'Head_Massage', 'High_Jump', 'Horse_Riding', 'Hula_Hoop', 'Javelin_Throw', 'Juggling_Balls', 'Jump_Rope', 'Kayaking', 'Knitting', 'Long_Jump', 'Lunges', 'Mopping_Floor', 'Playing_Cello', 'Playing_Flute', 'Playing_Guitar', 'Playing_Piano', 'Playing_Violin', 'Pole_Vault', 'Punch', 'Pull_Ups', 'Push_Ups', 'Rock_Climbing_Indoor', 'Rope_Climbing', 'Salsa_Spin', 'Shaving_Beard', 'Shotput', 'Skate_Boarding', 'Skiing', 'Skijet', 'Sky_Diving', 'Soccer_Juggling', 'Soccer_Penalty', 'Surfing', 'Swing', 'TaiChi', 'Tennis_Swing', 'Throw_Discus', 'Trampoline_Jumping', 'Typing', 'Volleyball_Spiking', 'Walking_With_Dog', 'Writing_On_Board'],
        'test':['Apply_Lipstick', 'Balance_Beam', 'Billiards', 'Blow_Dry_Hair', 'Fencing', 'Field_Hockey_Penalty', 'Front_Crawl', 'Hammering', 'Handstand_Pushups', 'Handstand_Walking', 'Horse_Race', 'Ice_Dancing', 'Jumping_Jack', 'Military_Parade', 'Mixing', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Daf', 'Playing_Dhol', 'Playing_Sitar', 'Playing_Tabla', 'Pommel_Horse', 'Rafting', 'Rowing', 'Still_Rings', 'Sumo_Wrestling', 'Table_Tennis_Shot', 'Uneven_Bars', 'Wall_Pushups', 'YoYo'],
        }
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset

        classes_fn = os.path.join(ANNO_PATH, 'classInd.txt')
        
        if subset == 'train':
            classes = sorted([k.replace('_', '') for k in UCF_ZSL['train']])
        elif subset == 'test':
            classes = sorted([k.replace('_', '') for k in UCF_ZSL['test']])
        elif subset == 'all':
            # https://arxiv.org/pdf/2003.01455.pdf; 
            _tmp = UCF_ZSL['test']
            _tmp.extend(UCF_ZSL['train'])
            classes = sorted([k.replace('_', '') for k in _tmp])
        else:
            raise NotImplementedError()
            
            
        # sanity all classes are in all_classes
        all_classes = [l.strip().split()[1] for l in open(classes_fn)]
        for k in classes:
            assert k in all_classes
                                                
        filenames = [ln.split('/')[-2:] for ln in glob.glob(os.path.join(f"{DATA_PATH}", "*","*.avi"))\
                     if ln.split('/')[-2] in classes]
            
        filenames = [os.path.join(p[0], p[1]) for p in filenames]
        label_names = [' '.join(re.findall('[A-Z][^A-Z]*', fn.replace('\\', '/').split('/')[0])).lower()
                      for fn in filenames] 
        
        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('ucf101', classes, wv_model)
        if isinstance(class_embeddings, list): # classes2embedding2 returns 2
            assert len(class_embeddings)==2
            label_names=class_embeddings[1] # words that fed to the word2vec
            class_embeddings=class_embeddings[0]
        
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
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )


class Kinetics(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=1,
                 wv_model=None,
                 classes2embedding=None,
                 ):

        # assert subset in ['train'] # for test use ucf01/hmdb51
        ROOT = os.path.join(f"{DATA_PATH}", f"{subset}")
        self.name = 'Kinetics-400'
        self.root = ROOT
        self.subset = subset
        
        classes = sorted(os.listdir(ROOT))
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'cache')
        CACHE_FILE = os.path.join(cache_dir, 'kinetics400', f"{subset}.txt")
        assert os.path.exists(CACHE_FILE), f"{CACHE_FILE} not found"
        # CACHE_FILE--> labels/videoname.avi,
        
        with open(CACHE_FILE, 'r') as f:
            files=json.loads(f.read())
         
        filenames = files
        label_names = [fn.split('/')[-2].replace('_', ' ') for fn in filenames] # words
        
        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('kinetics400', classes, wv_model)
        if isinstance(class_embeddings, list): # classes2embedding2 returns 2
            assert len(class_embeddings)==2
            label_names=class_embeddings[1] # words that fed to the word2vec
            class_embeddings=class_embeddings[0]

        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = classes
        
        super(Kinetics, self).__init__(
            return_video=True,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )

class Kinetics700(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=1,
                 wv_model=None,
                 classes2embedding=None,
                 ):

        # assert subset in ['train'] # for test use ucf01/hmdb51
        ROOT = os.path.join(f"{DATA_PATH}", f"{subset}")
        self.name = 'Kinetics-700'
        self.root = ROOT
        self.subset = subset
        
        classes = sorted(os.listdir(ROOT))
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'cache')
        CACHE_FILE = os.path.join(cache_dir, 'kinetics700', f"{subset}_663.txt")
        assert os.path.exists(CACHE_FILE), f"{CACHE_FILE} not found"
        # CACHE_FILE--> labels/videoname.avi,
        
        with open(CACHE_FILE, 'r') as f:
            files=json.loads(f.read())
         
        filenames = files
        label_names = [fn.split('/')[-2].replace('_', ' ') for fn in filenames] # words
        
        classes = list(np.unique(label_names))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('kinetics700', classes, wv_model)
        if isinstance(class_embeddings, list): # classes2embedding2 returns 2
            assert len(class_embeddings)==2
            label_names=class_embeddings[1] # words that fed to the word2vec
            class_embeddings=class_embeddings[0]

        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = classes
        
        super(Kinetics700, self).__init__(
            return_video=True,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )

# DATA_PATH='/mnt/PS6T/datasets/Video/RareAct/'

class RareAct(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset=1,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=1,
                 wv_model=None,
                 classes2embedding=None,
                 ):

        """ subset information
        1: Positive. 
        2: Hard negative (only verb is right).
        3: Hard negative (only noun is right). 
        4: Hard negative (Both verb and noun are valid but verb is not applied to noun). 
        0: Negative.
        
        following https://proceedings.neurips.cc/paper/2021/file/d6539d3b57159babf6a72e106beb45bd-Paper.pdf
        we are interested on positive only for ZSL.
        """
        
        assert subset == 1, NotImplementedError()
        ROOT = os.path.join(f"{DATA_PATH}", 'rareact_processed')
        files = [fn.split('/')[-1:][0] for fn in glob.glob(os.path.join(f"{ROOT}", "*.avi")) if int(fn[-6:-5]) == subset]

        self.name = 'RareAct'
        self.root = ROOT
        
        # fetching meta data
        label_names = []
        # labels, verbs, nouns, label_names = [], [], [], []
        for fn in files:
            _fn = fn.split('_')
            # nouns.append(_fn[-3])
            # verbs.append(_fn[-4])
            # labels.append(int(_fn[-5]))
            label_names.append(_fn[-4] + ' ' + _fn[-3])
            
        classes = sorted(list(np.unique(label_names)))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('rareact', classes, wv_model)
        if isinstance(class_embeddings, list): # classes2embedding2 returns 2
            assert len(class_embeddings)==2
            label_names=class_embeddings[1] # words that fed to the word2vec
            class_embeddings=class_embeddings[0]
        
        filenames = files

        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = classes
        
        super(RareAct, self).__init__(
            return_video=True,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )


class Mimetics(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset='mimetics50', # mimetics50, mimetics10
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 mode='clip',
                 clips_per_video=1,
                 wv_model=None,
                 classes2embedding=None,
                 ):

        """ loads only the validation set """
        from datasets.loader.mimetics import mimetics10, mimetics50
        ROOT = os.path.join(f"{DATA_PATH}", "videos")
        self.name = 'Mimetics dataset'
        self.root = ROOT
        self.subset = subset
        
        classes = sorted(eval(subset))
        
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'mimetics', "val.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        else:
            raise FileNotFoundError(f"{CACHE_FILE}")
            

        filenames = [fn for fn in files if fn.split('/')[-2] in classes]
        label_names = [fn.split('/')[-2].replace('_', ' ') for fn in filenames] # words
        
        
        classes = sorted(list(np.unique(label_names)))
        labels = [classes.index(ll) for ll in label_names]
        class_embeddings = classes2embedding('kinetics400', classes, wv_model) # class names are same as kinetics400
        
        self.num_classes = len(classes)
        self.num_videos = len(filenames)
        self.classes = classes
        
        super(Mimetics, self).__init__(
            return_video=True,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=True,
            return_index=True,
            return_tokens=True,
            labels=labels,
            label_names=label_names,
            class_embeddings=class_embeddings,
            mode=mode,
            clips_per_video=clips_per_video,
        )

