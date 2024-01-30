# VSSL OoD Validation

### Environment Setup
Our codes are based on PyTorch. The additional dependencies can be found `requirements.txt`. You can create an environment as `conda create --name vssl-ood --file requirements.txt`


### Pretrained model weights

The VSSL pretraining weights can be downloaded from this [link](https://drive.google.com/drive/folders/1Hk1mXjwiTKUxO_Cd4gnwUE5fxr5u3eMS?usp=sharing).

| **Methods**                     | **Weights**     |
|---------------------------------|--------------|
|v-SimCLR| `VideoSimCLR_kinetics400.pth.tar`|
|v-MOCO| `VideoMOCOv3_kinetics400.pth.tar`| -->
|v-BYOL| `VideoBYOL_kinetics400.pth.tar`|
|v-SimSiam| `VideoSimSiam_kinetics400.pth.tar`|
|v-DINO| `VideoDINO_kinetics400.pth.tar`|
|v-MAE| `VideoMAE_kinetics400.pth.tar`|
|v-Supervised| `VideoSupervised_kinetics400.pth.tar`| 

### Dataset

- Following, we list the sources of the datasets used in this work. We follow standard/official instructions for usage, please see the documentation of the respective datasets for details. 
- Once the datasets are downloaded please update the paths in `tools/paths.py`. 
- Please download the `cache.zip` for eval. datasets from this [link](https://drive.google.com/file/d/1hn_DiWScgr0aYdbd8_PgIfcsixWFsBE6/view?usp=sharing) and unzip inside `datasets/`. 


| **Datasets** | **License** | **Link** |
|----------------|--------------|--------------------|
CharadesEgo | License for Non-Commercial Use| https://prior.allenai.org/projects/charades-ego
Moments-in-Time-v2 | License for Non-Commercial Use| http://moments.csail.mit.edu/
Kinetics | CC BY 4.0 | https://www.deepmind.com/open-source/kinetics
HMDB51 | CC BY 4.0 | https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
ToyBox | CC BY 4.0 | https://aivaslab.github.io/toybox/
Mimetics | Open access | https://europe.naverlabs.com/research/computer-vision/mimetics/
UCF101 | Open access | https://www.crcv.ucf.edu/data/UCF101.php
TinyVirat-v2 | Open access | https://www.crcv.ucf.edu/tiny-actions-challenge-cvpr2021/#tabtwo
COIL100 | Open access | https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
STL-10 | Open access | https://cs.stanford.edu/~acoates/stl10/
ActorShift | MIT | https://uvaauas.figshare.com/articles/dataset/ActorShift_zip/19387046
Sims4Action| MIT | https://github.com/aroitberg/sims4action
RareAct| Apache | https://github.com/antoine77340/RareAct

### Details of OoD experiments

| **#** | **Distribution Shift**                     | **In-distribution**     | **Out-of-distribution**        | **#Classes** | **#Samples** (train/InD test/OoD test)      |
|-------|-------------------------------------------|-------------|----------------|--------------|--------------------|
| 1.    | Context shift (10 classes)                | Kinetics400 | Mimetics10     | 10           | 5930/494/136       |
| 2.    | Context shift (50 classes)                | Kinetics400 | Mimetics50     | 50           | 34K/2481/713        |
| 3.    | Viewpoint shift (egocentric)              | CharadesEgo | CharadesEgo    | 157          | 34K/9386/9145       |
| 4.    | Viewpoint shift (surveillance+low resolution)            | MiT-v2      | TinyVirat-v2   | 14           | 41K/1400/2644       |
| 5.    | Actor shift (animal)                       | Kinetics400 | ActorShift     | 7            | 15K/1018/165        |
| 6.    | Viewpoint + Actor shift (top-down+synthetic)| MiTv2      | Sims4Action    | 6            | 19K/600/950         |
| 7.    | Source shift (UCF/HMDB)                    | UCF         | HMDB           | 17           | 1877/746/510       |
| 8.    | Source shift (HMDB/UCF)                    | HMDB        | UCF            | 17           | 1190/510/746       |
| 9.    | Zero-shot (K400/UCF)                       | Kinetics400 | UCF            | 400/31       | 240K/20K/3965       |
| 10.   | Zero-shot (K400/HMDB)                      | Kinetics400 | HMDB           | 400/22       | 240K/20K/3288       |
| 11.   | Zero-shot (K400/RareAct)                   | Kinetics400 | RareAct        | 400/149      | 240K/20K/1961       |
| 12.   | Zero-shot (K700/UCF)                       | Kinetics700 | UCF            | 663/101      | 480K/-/13K          |
| 13.   | Zero-shot (K700/HMDB)                      | Kinetics700 | HMDB           | 663/51       | 480K/-/6.7K         |
| 14.   | Zero-shot (K700/RareAct)                   | Kinetics700 | RareAct        | 663/149      | 480K/-/1961         |
| 15.   | Open-set (K400/UCF)                        | Kinetics400 | UCF            | 400/31       | 240K/20K/3965       |
| 16.   | Open-set (K400/HMDB)                       | Kinetics400 | HMDB           | 400/22       | 240K/20K/3288       |
| 17.   | Open-set (U101/HMDB)                       | UCF101      | HMDB           | 101/34       | 9537/3783/4366     |


### OoD evaluation
- To run these jobs, please go inside `src/jobs`. 
- You have to modify the `.sh` file based on your server configuration, the provided file can be treated just as a reference. 
- All finetuning jobs require 4 V100 32GB GPUs or something similar. 
- The given scripts are meant to be used for SLURM-based server.
- In the following examples we use the v-BYOL weights; same can be done for the other pretraining weights as well.

### Finetune evaluation

#### Context shift (1, 2)

```
bash launch.sh finetune-ood byol_k400_mimetics10.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh finetune-ood byol_k400_mimetics50.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
```

#### Viewpoint (3, 4, 5)

```
bash launch.sh finetune-ood-cego byol_train3rd_test1st.yaml charadesego VideoBYOL_kinetics400.pth.tar
bash launch.sh finetune-ood-mit-tiny byol_mit_tiny_v2.yaml mitv2 VideoBYOL_kinetics400.pth.tar
bash launch.sh finetune-ood byol_k700_actor_shift.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
```
#### Actor shift (5, 6)

```
bash launch.sh finetune-ood byol_k700_actor_shift.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh finetune-ood byol_mit_sims4action.yaml mitv2 VideoBYOL_kinetics400.pth.tar
```

#### Source shift (7, 8)

```
bash launch.sh finetune-ood byol_hmdb_ucf.yaml hmdb51 VideoBYOL_kinetics400.pth.tar
bash launch.sh finetune-ood byol_ucf_hmdb.yaml ucf101 VideoBYOL_kinetics400.pth.tar
```

#### Zeroshot (9-11, 12-14)

```
bash launch.sh zsl-finetune byol_zsl.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh zsl-finetune byol_zsl.yaml kinetics700 VideoBYOL_kinetics700.pth.tar
```

#### Openset (15, 16, 17)

```
bash launch.sh openset byol_ft_u101_hmdb_dear.yaml ucf101 VideoBYOL_kinetics400.pth.tar
bash launch.sh openset byol_ft_k400_hmdb_dear.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh openset byol_ft_k400_ucf_dear.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
```

### Linear evaluation


#### Context shift (1, 2)

```
bash launch.sh linear-ood byol_k400_mimetics10.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh linear-ood byol_k400_mimetics50.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
```

#### Viewpoint (3, 4, 5)

```
bash launch.sh linear-ood-cego byol_train3rd_test1st.yaml charadesego VideoBYOL_kinetics400.pth.tar
bash launch.sh linear-ood-mit-tiny byol_mit_tiny_v2.yaml mitv2 VideoBYOL_kinetics400.pth.tar
bash launch.sh linear-ood byol_k700_actor_shift.yaml kinetics400 VideoBYOL_kinetics400.pth.tar

# note linear-ood-mit-tiny: originally I extracted fixed features separately and ran fc tuning, this strategy saves compute time. But, to be consistent with other evaluation scripts this code loads the videos and do fc tuning in an usual training loop. A slight performance difference can be expected and I expect the current setup will result better performance.
```
#### Actor shift (5, 6)

```
bash launch.sh linear-ood byol_k700_actor_shift.yaml kinetics400 VideoBYOL_kinetics400.pth.tar
bash launch.sh linear-ood byol_mit_sims4action.yaml mitv2 VideoBYOL_kinetics400.pth.tar
```

#### Source shift (7, 8)

```
bash launch.sh linear-ood byol_hmdb_ucf.yaml hmdb51 VideoBYOL_kinetics400.pth.tar
bash launch.sh linear-ood byol_ucf_hmdb.yaml ucf101 VideoBYOL_kinetics400.pth.tar
```