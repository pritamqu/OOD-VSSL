import os
import random
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict

def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


class VideoDataset(data.Dataset):
    def __init__(self,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_clip_duration=0.5,
                 video_fps=16,
                 video_transform=None,
                 return_labels=False,
                 return_tokens=False,
                 labels=None,
                 label_names=None,
                 class_embeddings=None,
                 return_index=False,
                 mode='clip', # video
                 clips_per_video=1,
                 ):
        super(VideoDataset, self).__init__()

        self.num_samples = 0

        self.return_video = return_video
        self.video_root = video_root
        if return_video:
            self.video_fns = chararray(video_fns)
            self.num_samples = self.video_fns.shape[0]
        self.video_fps = video_fps
        self.video_transform = video_transform
        self.return_labels = return_labels
        self.return_tokens = return_tokens
        if return_labels:
            self.labels = np.array(labels)
            self.labels = self.labels.astype(np.int64)
        if return_tokens:
            self.label_names = label_names
            self.class_embeddings = class_embeddings
        self.return_index = return_index
        self.video_clip_duration = video_clip_duration
        self.clips_per_video = clips_per_video
        self.mode = mode               

    def _load_sample(self, sample_idx):
        """ it loads a sample video to a container"""
        video_ctr = None
        if self.return_video:
            video_fn = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
            video_ctr = av_open(video_fn)

        return video_ctr

    def __getitem__(self, index):
        
        ########### just one clip for regular use
        #########################################
        
        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                video_ctr = self._load_sample(sample_idx)
                v_ss, v_dur = self._sample_snippet(video_ctr)   
                sample = self._get_clip(sample_idx, video_ctr, v_ss, video_clip_duration=v_dur)
                if sample is None:
                    return self[(index+1) % len(self)]

                return sample
            except Exception:
                return self[(index+1) % len(self)]
            
        ########### return clips_per_video number of clips from whole video
        ###################################################################

        elif self.mode == 'video':
            video_ctr = self._load_sample(index)

            # Load entire video
            vs, vf = self._get_time_lims(video_ctr)
            if self.return_video:
                start_time = vs
                final_time = vf
                if final_time <= start_time:
                    final_time = start_time + self.video_clip_duration
                    
            video_dur = final_time - start_time
            sample = self._get_clip(index, video_ctr, start_time, video_clip_duration=video_dur)

            # Split video into overlapping chunks
            chunks = defaultdict(list)

            if self.return_video:
                
                if len(sample['frames'].shape)==5: # if 3 crops are added
                    nf = sample['frames'].shape[2]
                    chunk_size = int(self.video_clip_duration * self.video_fps)
                    if chunk_size >= nf:
                        # chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                        chunks['frames'] = torch.cat([sample['frames'] for _ in range(self.clips_per_video)], dim=0)
                    else:
                        timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                        # chunks['frames'] = torch.stack([sample['frames'][:, :, ss:ss+chunk_size] for ss in timestamps])
                        chunks['frames'] = torch.cat([sample['frames'][:, :, ss:ss+chunk_size] for ss in timestamps], dim=0)
                
                else:
                    nf = sample['frames'].shape[1]
                    chunk_size = int(self.video_clip_duration * self.video_fps)
                    if chunk_size >= nf:
                        chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                    else:
                        timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                        chunks['frames'] = torch.stack([sample['frames'][:, ss:ss+chunk_size] for ss in timestamps])
                    
            if self.return_labels:
                chunks['label'] = sample['label']
                
            if self.return_tokens:
                # chunks['label_names'] = sample['label_names']
                chunks['class_embeddings'] = sample['class_embeddings']

            if self.return_index:
                # ts = torch.from_numpy(np.linspace(start_time, final_time-self.audio_clip_duration, self.clips_per_video))
                # chunks['index'] = torch.stack([torch.tensor(sample['index']).repeat(self.clips_per_video), ts.float()], dim=1)
                chunks['index'] = sample['index']
                
            return chunks
        

    def __len__(self):
        if self.mode == 'clip' or self.mode == 'two_clips':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n - Num classes: {}\n".format(
            self.name, self.root, self.subset, self.num_videos, self.num_videos * self.clips_per_video, self.num_classes)
        if self.return_video:
            desc += " - Example video: {}/{}\n".format(self.video_root, self.video_fns[0].decode())
        return desc

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

    def _sample_snippet(self, video_ctr):
        video_st, video_ft = self._get_time_lims(video_ctr)
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

        sample = {}
        if self.return_video:
            frames, fps, start_time = av_load_video(
                video_ctr,
                video_fps=self.video_fps,
                start_time=video_start_time,
                duration=video_clip_duration,
            )
            if self.video_transform is not None:
                frames = self.video_transform(frames)

            sample['frames'] = frames

        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl
                
        if self.return_tokens:
            
            lbl = self.labels[clip_idx]
            # sample['label_names'] = self.label_names[lbl]
            sample['class_embeddings'] = self.class_embeddings[lbl]

        if self.return_index:
            sample['index'] = clip_idx

        return sample
    
    
###############################################################################

import av
import math
from scipy.interpolate import interp1d

av.logging.set_level(0)

def av_open(inpt):
    try:
        container = av.open(inpt)
    except:
        container = av.open(inpt, metadata_errors="ignore")
    return container

# the original code from AVID
def av_load_video(container, video_fps=None, start_time=0, duration=None):
    video_stream = container.streams.video[0]
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur
    _fps = video_stream.average_rate

    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = _ff - start_time

    # Figure out which frames to decode
    outp_times = [t for t in np.arange(start_time, min(start_time + duration - 0.5/_fps, _ff), 1./video_fps)][:int(duration*video_fps)]
    outp_vframes = [int((t - _ss) * _fps) for t in outp_times]
    start_time = outp_vframes[0] / float(_fps)

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    frames = []
    for frame in container.decode(video=0):
        if len(frames) == len(outp_vframes):
            break   # All frames have been decoded
        frame_no = frame.pts * frame.time_base * _fps
        if frame_no < outp_vframes[len(frames)]:
            continue    # Not the frame we want

        # Decode
        pil_img = frame.to_image()
        while frame_no >= outp_vframes[len(frames)]:
            frames += [pil_img]
            if len(frames) == len(outp_vframes):
                break   # All frames have been decoded

    return frames, video_fps, start_time

