#!/usr/bin/env python3
import os
import glob
import numpy as np
import random
import ffmpeg
import json
import pickle
import torch.utils.data as data
from collections import defaultdict
import torch

class FeatDataset(data.Dataset):
    def __init__(self,
                 root,
                 split,
                 ):
        super(FeatDataset, self).__init__()

        print(f"loading files from {root}")
        
        self.data = self._load_data(root, split)
        
    def _load_data(self, root, split):
        # kinetics400_strong3crop_aug_mode_train_train_feats_29.pkl
        if split == 'train':
            """ 2 seconds of 10 clips per video randomly sampled at 8 FPS with aug of color jitter + crop + horizontal flip
            """
            pkl_files = [fn for fn in glob.glob(os.path.join(f"{root}", "*.pkl")) if '_'.join(fn.split('_')[-3:-1]) == 'train_feats']
            
        else:
            """ 2 seconds of 5 clips x3 crops per video uniformly sampled at 8 FPS with no aug
            """
            pkl_files = [fn for fn in glob.glob(os.path.join(f"{root}", "*.pkl")) if '_'.join(fn.split('_')[-3:-1]) != 'train_feats']

        data = self._accumulate_mul_pkl(pkl_files)
        
        return data
    
    def _accumulate_mul_pkl(self, all_files):
        total_features = defaultdict(list)
        keys = ['vis_feat_bank', 'feature_labels', 'feature_indexs']
        for ff in all_files:
            _features = pickle.load(open(ff,'rb'))
            for k in keys:
                total_features[k].append(_features[k])
                
        total_features['vis_feat_bank'] = torch.vstack(total_features['vis_feat_bank'])
        total_features['feature_labels'] = torch.hstack(total_features['feature_labels'])
        total_features['feature_indexs'] = torch.hstack(total_features['feature_indexs'])
            
        return total_features

    def __getitem__(self, index):
        
        return {'frames': self.data['vis_feat_bank'][index], 
                'label': self.data['feature_labels'][index], 
                'index': self.data['feature_indexs'][index], 
                }
    
    def __len__(self):
        return len(self.data['vis_feat_bank'])
    
    
class FeatDataset_ML(data.Dataset):
    """ multi-label dataset """
    def __init__(self,
                 root,
                 split,
                 ):
        super(FeatDataset_ML, self).__init__()

        print(f"loading files from {root}")
        
        self.data = self._load_data(root, split)
        
    def _load_data(self, root, split):
        
        if split == 'train':
            # kinetics400_strong3crop_aug_mode_train_train_feats_29.pkl
            """ 2 seconds of 10 clips per video randomly sampled at 8 FPS with aug of color jitter + crop + horizontal flip
            """
            pkl_files = [fn for fn in glob.glob(os.path.join(f"{root}", "*.pkl")) if '_'.join(fn.split('_')[-4:-1]) == 'train_train_feats']
            
        elif split == 'val':
            # charadesego_strong3crop_aug_mode_val_val_feats_11
            """ 2 seconds of 5 clips x3 crops per video uniformly sampled at 8 FPS with no aug
            """
            pkl_files = [fn for fn in glob.glob(os.path.join(f"{root}", "*.pkl")) if '_'.join(fn.split('_')[-4:-1]) == 'val_val_feats']
        
        elif split == 'ood_val':
            # charadesego_strong3crop_aug_mode_val_val_feats_11
            """ 2 seconds of 5 clips x3 crops per video uniformly sampled at 8 FPS with no aug
            """
            pkl_files = [fn for fn in glob.glob(os.path.join(f"{root}", "*.pkl")) if '_'.join(fn.split('_')[-5:-1]) == 'val_ood_val_feats']
        else:
            raise NotImplementedError()

        # print("************ taking a subset of data for debug ************")
        # data = self._accumulate_mul_pkl(pkl_files[:10])
        data = self._accumulate_mul_pkl(pkl_files)
        
        return data
    
    def _accumulate_mul_pkl(self, all_files):
        total_features = defaultdict(list)
        keys = ['vis_feat_bank', 'feature_labels', 'feature_indexs']
        for ff in all_files:
            _features = pickle.load(open(ff,'rb'))
            for k in keys:
                total_features[k].append(_features[k])
                
        total_features['vis_feat_bank'] = torch.vstack(total_features['vis_feat_bank'])
        total_features['feature_labels'] = torch.vstack(total_features['feature_labels'])
        total_features['feature_indexs'] = torch.hstack(total_features['feature_indexs'])
            
        return total_features

    def __getitem__(self, index):
        
        return {'frames': self.data['vis_feat_bank'][index], 
                'label': self.data['feature_labels'][index], 
                'index': self.data['feature_indexs'][index], 
                }
    
    def __len__(self):
        return len(self.data['vis_feat_bank'])
    
    