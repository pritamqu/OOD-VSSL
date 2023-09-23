import os
from datasets.loader.backend.video_db import VideoDataset


# from tools.paths import my_paths
# DATA_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'HMDB-51'),
# ANNO_PATH = os.path.join(my_paths('local', 'hmdb51')[-1], 'testTrainMulti_7030_splits'),
# subset = train-split01

class HMDB(VideoDataset):
    def __init__(self, 
                  DATA_PATH,
                  ANNO_PATH,
                  subset,
                  return_video=True,
                  video_clip_duration=1.,
                  video_fps=25.,
                  video_transform=None,
                  return_audio=False,
                  return_labels=False,
                  return_index=True,
                  mode='clip',
                  clips_per_video=20,
                  ):
        assert return_audio is False
        self.name = 'HMDB-51'
        self.root = DATA_PATH
        self.subset = subset

        # Get filenames
        classes = sorted(os.listdir(DATA_PATH))
        subset, split = subset.split('-')
        subset_id = {'train': '1', 'test': '2'}[subset]
        filenames, labels = [], []
        for cls in classes:
            for ln in open(os.path.join(f'{ANNO_PATH}', f'{cls}_test_{split}.txt')): 
                fn, ss = ln.strip().split()
                if ss == subset_id:
                    filenames += [os.path.join(f"{cls}", f"{fn}")]
                    labels += [classes.index(cls)]

        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(HMDB, self).__init__(
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=return_labels,
            return_index=return_index,
            labels=labels,
            mode=mode,
            clips_per_video=clips_per_video,
        )
