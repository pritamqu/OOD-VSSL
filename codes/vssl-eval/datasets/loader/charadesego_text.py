
# reference https://github.com/showlab/EgoVLP/blob/main/data_loader/CharadesEgo_dataset.py


import os
import sys
import csv
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import torch
import random
import numpy as np
from collections import defaultdict
import ffmpeg
from joblib import Parallel, delayed


# from base.base_dataset import TextVideoDataset
try:
    from datasets.loader.backend import av_wrappers
except:
    from backend import av_wrappers
    
# try:
#     from transforms import init_transform_dict, init_video_transform_dict
# except:
#     pass

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
    valid_files = ['/'.join(vid_paths[i].split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files


class CharadesEgoText(torch.utils.data.Dataset):
    def __init__(self,
                 # return_video=True,
                 root=None,
                 # video_fns=None,
                 video_clip_duration=0.5,
                 subset=None,
                 split='train',
                 video_fps=16,
                 video_transform=None,
                 # return_labels=False,
                 # labels=None,
                 # return_index=False,
                 mode='clip', # video
                 clips_per_video=1,
                 subsample=1, # take a random fraction of data
                 ):
        super(CharadesEgoText, self).__init__()
    
        self.name = f"CharadesEgo -{subset}"
        self.num_classes = 157
        self.video_clip_duration = video_clip_duration
        self.video_transform = video_transform
        self.video_fps = video_fps
        self.clips_per_video = clips_per_video
        if subset in ['train_1st', 'train_3rd']:
            split='train'
        elif subset in ['test_1st', 'test_3rd']:
            split='val'
        else:
            assert subset==None, f'Unknown subset {subset}'
        self.split=split
        self.mode = mode
        self.data_dir =  os.path.join(root, 'CharadesEgo_v1_480')
        self.meta_dir = os.path.join(root, 'CharadesEgo', 'video-language')
        self._load_metadata(subset)
        self.num_samples = len(self.metadata)
    
    def __repr__(self):
        desc = "{}\n - Root: {}\n - Split: {}\n - Num videos: {}\n - Num samples: {}\n - Num classes: {}\n".format(
            self.name, self.data_dir, self.split, self.num_samples, self.num_samples * self.clips_per_video, self.num_classes)
        
        return desc
    
    def __len__(self):
        if self.mode == 'clip':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples
        
    def _load_metadata(self, subset):
        split_files = {
            'train_1st': 'metadata_train_1st.csv',
            'train_3rd': 'metadata_train_3rd.csv', # ood train
            'test_1st': 'metadata_test_1st.csv', # ood test
            'test_3rd': 'metadata_test_3rd.csv', # iid test
            # 'train': 'metadata_train.csv',
            # 'val': 'CharadesEgo_v1_test_only1st.csv',
            # 'test': 'CharadesEgo_v1_test_only1st.csv'
        }
        target_split_fp = split_files[subset]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), delimiter='\t')
        
        # if self.split == 'train':
        #     metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), delimiter='\t')
        # else:
        #     metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        # if self.subsample < 1:
        #     metadata = metadata.sample(frac=self.subsample)

        self.metadata = metadata
        
        if subset == 'test_1st':
            self.label = self._parse_charades_csv(os.path.join(os.path.dirname(self.meta_dir), "CharadesEgo_v1_test_only1st.csv"))
        elif subset == 'test_3rd':
            self.label = self._parse_charades_csv(os.path.join(os.path.dirname(self.meta_dir), "CharadesEgo_v1_test_only3rd.csv"))
        else:
            self.label = None

    def _parse_charades_csv(self, filename):
        labels = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions = row['actions']
                if actions == '':
                    actions = []
                else:
                    actions = [a.split(' ') for a in actions.split(';')]
                    actions = [{'class': x, 'start': float(
                        y), 'end': float(z)} for x, y, z in actions]
                labels[vid] = actions
        return labels

    def _get_video_path(self, sample):
        rel_video_fp = sample['id'] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # if self.split in ['val', 'test']:
        #     return sample[6]
        # else:
        return sample['narration']

    def _cls2int(self, x):
        return int(x[1:])

    def __getitem__(self, item):
        if self.mode == 'video':
            return self._get_val(item)
        else:
            return self._get_train(item)

    def _get_time_lims(self, video_ctr):
        video_st, video_ft = None, None
        if video_ctr is not None:
            video_stream = video_ctr.streams.video[0]
            tbase = video_stream.time_base
            ## TODO: recheck this
            # audio_stream is av.audio.codeccontext.AudioCodecContext object, does not have start time
            video_st = video_stream.start_time * tbase
            # video_st = 0
            video_dur = video_stream.duration * tbase
            video_ft = video_st + video_dur

        return video_st, video_ft
    
    def _sample_snippet(self, video_ctr, start_sec, end_sec):
        video_st, video_ft = self._get_time_lims(video_ctr)
        video_st = np.round(float(video_st), 2)
        video_ft = np.round(float(video_ft), 2)
        # sanity
        if start_sec<video_st:
            # print(f"DONOT Satisfy {start_sec}>={video_st}")
            start_sec=video_st
        if end_sec>video_ft:
            # print(f"DONOT Satisfy {end_sec}<={video_ft}")
            end_sec=video_ft
        # assert start_sec>=video_st and end_sec<=video_ft, f"DONOT Satisfy {start_sec}>={video_st} {end_sec}<={video_ft}" # sanity
        
        video_st, video_ft = start_sec, end_sec
        video_duration = video_ft - video_st
        if self.video_clip_duration > video_duration:
            return 0., video_duration
        else:
            min_d, max_d = self.video_clip_duration, min(self.video_clip_duration, video_duration)
            duration = random.uniform(min_d, max_d)
            sample_ss_v = random.uniform(video_st, video_ft - duration)
            return sample_ss_v, duration
        
    def _get_clip(self, clip_idx, video_ctr, video_start_time, video_clip_duration=None):
        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration

        # sample = {}
        # if self.return_video:
        frames, fps, start_time = av_wrappers.av_load_video(
            video_ctr,
            video_fps=self.video_fps,
            start_time=video_start_time,
            duration=video_clip_duration,
        )
        if self.video_transform is not None:
            frames = self.video_transform(frames)

        return frames
        # sample['frames'] = frames

        # if self.return_labels:
        #     lbl = self.labels[clip_idx]
        #     if isinstance(lbl, np.ndarray):
        #         sample['label'] = torch.from_numpy(lbl)
        #     else:
        #         sample['label'] = lbl

        # if self.return_index:
        #     sample['index'] = clip_idx

        # return sample
    
    def _load_sample(self, video_fn):
        """ it loads a sample video to a container"""
        video_ctr = None
        # if self.return_video:
        # video_fn = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
        video_ctr = av_wrappers.av_open(video_fn)

        return video_ctr
    
    def _get_train(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        start_sec, end_sec = sample['t_start'],  sample['t_end']
        video_ctr = self._load_sample(video_fp)
        v_ss, v_dur = self._sample_snippet(video_ctr, start_sec, end_sec) 
        frames = self._get_clip(item, video_ctr, v_ss, video_clip_duration=v_dur)

        data = {'frames': frames, 'text': caption, 'label': sample['cls']}

        return data

    def _get_val(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        # construct label
        label = self.label[sample['id']]
        target = torch.IntTensor(157).zero_()
        for x in label:
            target[self._cls2int(x['class'])] = 1
            
        start_sec, end_sec = sample['t_start'],  sample['t_end']
        video_ctr = self._load_sample(video_fp)
        video_dur = end_sec - start_sec
        frames = self._get_clip(item, video_ctr, start_sec, video_clip_duration=video_dur)
            
        # Split video into overlapping chunks
        data = defaultdict(list)
        
            
        if len(frames.shape)==5: # if 3 crops are added
            nf = frames.shape[2]
            chunk_size = int(self.video_clip_duration * self.video_fps)
            if chunk_size >= nf:
                # chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                data['frames'] = torch.cat([frames for _ in range(self.clips_per_video)], dim=0)
            else:
                timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                data['frames'] = torch.cat([frames[:, :, ss:ss+chunk_size] for ss in timestamps], dim=0)
        
        else:
            nf = frames.shape[1]
            chunk_size = int(self.video_clip_duration * self.video_fps)
            if chunk_size >= nf:
                data['frames'] = torch.stack([frames for _ in range(self.clips_per_video)])
            else:
                timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                data['frames'] = torch.stack([frames[:, ss:ss+chunk_size] for ss in timestamps])
                        
        data['text'] = caption
        data['label'] = target
        data['id'] = sample['id']
        
        return data
