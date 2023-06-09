# Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

<h3 align="center">
Under review.
</h3>
<h3 align="center">
<a href="https://www.pritamsarkar.com">Pritam Sarkar</a>
&nbsp;
Ahmad Beirami
&nbsp;
Ali Etemad
</h3>
<h3 align="center"> 
<a href="https://arxiv.org/abs/2306.02014">[Paper]</a>
<a href="https://pritamqu.github.io/OOD-VSSL/"> [Website]</a>
</h3>

##### Codes will be released here soon. You may follow this repo to receive updates.


### A comprehensive out-of-distribution test bed for VSSL.

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


### Video self-supervised learning methods studied in this work.

| ![simclr](/docs/assets/images/simclr.png) | ![moco](/docs/assets/images/moco.png) | ![mae](/docs/assets/images/mae.png) |
|:--:|:--:|:--:|
| **(a) v-SimCLR** | **(b) v-MOCO** | **(c) v-MAE** |
| ![byol](/docs/assets/images/byol.png) | ![simsiam](/docs/assets/images/simsiam.png) | ![dino](/docs/assets/images/dino.png) |
| **(d) v-BYOL** | **(e) v-SimSiam** | **(f) v-DINO** |

A simplified version of the video self-supervised methods are presented.




### Citation
If you find this repository useful, please consider giving a star :star: and citation using the given BibTeX entry:

```
@misc{sarkar2023ood,
      title={Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts}, 
      author={Pritam Sarkar and Ahmad Beirami and Ali Etemad},
      year={2023},
      eprint={2306.02014},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Contact me
You may directly contact me at <pritam.sarkar@queensu.ca> or connect with me on [LinkedIn](https://www.linkedin.com/in/sarkarpritam/).
I am **looking for internship** opportunity in related areas; if you have an opening, please feel free to reach out to me.
