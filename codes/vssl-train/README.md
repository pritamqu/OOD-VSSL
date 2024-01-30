
# VSSL Pretraining

### Methods
The following VSSL models are available:

- [x] v-SimCLR
- [x] v-MoCo
- [x] v-BYOL
- [x] v-SimSiam
- [x] v-DINO
- [x] v-MAE


### Environment Setup
Our codes are based on PyTorch. The additional dependencies can be found `requirements.txt`. You can create an environment as `conda create --name vssl-ood --file requirements.txt`


### Datasets

- Please make sure to keep the datasets in their respective directories and change the path in `/tools/paths` accordingly. 
- To download Kinetics400, please check the official website https://www.deepmind.com/open-source/kinetics.
- You can follow instruction given in this link to process the data: https://github.com/cvdfoundation/kinetics-dataset.
- Note: our codes are optimized for faster dataloading using `.avi` videos, which may require additional processing using `ffmpeg`. We provide a sample code in `datasets/process.py` to show an example how this can be done. A minor modifications might be required based on your directory/download setup.
- Please download the `cache.zip` from this [link](https://drive.google.com/file/d/1hn_DiWScgr0aYdbd8_PgIfcsixWFsBE6/view?usp=sharing) and unzip inside `datasets/`. These cache files will be used by the dataloaders. 
- After unzipping, the directory structure should be `datasets/cache/kinetics400/` and inside there should be `train.txt` and `val.txt`. we could not directly share the cache files with the code as it's more than the allowed file size in GitHub.


### Additional notes

- The following submission scripts (`src/jobs`) are meant to use for SLURM-based servers. You have to modify them based on your server setup, i.e., changing paths, account, partitions, cpu, gpu etc. 
- We perform multi-node multi-gpu training using PyTorch. The default setup would require 8 V100 GPUs (32GB) or similar. 
- If you are not familiar with pytorch distributed training please check official examples from PyTorch: https://pytorch.org/tutorials/beginner/dist_overview.html
- If you are not familiar with SLURM, please see SLURM documentation https://slurm.schedmd.com/documentation.html
- The codes are tested on a linux based server. If you encounter any error please let us know.

### VSSL Pretraining

go inside `src/jobs` and run the following commands to submit pretraining jobs.

```
sbatch --nodes 2 --ntasks 2 main_byol.sh byol.yaml kinetics400
sbatch --nodes 2 --ntasks 2 main_simsiam.sh simsiam.yaml kinetics400
sbatch --nodes 2 --ntasks 2 main_dino.sh mae.yaml kinetics400
sbatch --nodes 2 --ntasks 2 main_moco.sh mae.yaml kinetics400
sbatch --nodes 2 --ntasks 2 main_simclr.sh mae.yaml kinetics400
sbatch --nodes 2 --ntasks 2 main_mae.sh mae.yaml kinetics400
```

