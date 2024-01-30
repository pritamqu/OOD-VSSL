# Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

<h3 align="center">
NeurIPS 2023 (Spotlight)
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


### Availability

The following items are available in the repo, please go inside the sub-dirs to find the detailed documentations. 

- [x] VSSL evaluation codes: [vssl-eval](/codes/vssl-eval/README.md)
- [x] VSSL pretrained model: [link](https://drive.google.com/drive/folders/1Hk1mXjwiTKUxO_Cd4gnwUE5fxr5u3eMS?usp=sharing).
- [x] VSSL pretraining codes: [vssl-train](/codes/vssl-train/README.md)
- [ ] VSSL finetuned model: [link](/README.md).


### Real-world distribution shifts

![OOD-VSSL](/docs/assets/images/ood_vssl.png)

A simplified illustration of real-world distribution shifts that are studied in this work.


### Video self-supervised learning methods

| ![simclr](/docs/assets/images/simclr.png) | ![moco](/docs/assets/images/moco.png) | ![mae](/docs/assets/images/mae.png) |
|:--:|:--:|:--:|
| **(a) v-SimCLR** | **(b) v-MOCO** | **(c) v-MAE** |
| ![byol](/docs/assets/images/byol.png) | ![simsiam](/docs/assets/images/simsiam.png) | ![dino](/docs/assets/images/dino.png) |
| **(d) v-BYOL** | **(e) v-SimSiam** | **(f) v-DINO** |

A simplified version of the video self-supervised methods that are studied in this work.


### Disclaimer

This repo is currently under initial development phase and we plan to improve it in several aspects, including adding more documentation, adding proper acknowledgements to the references that are used to create this repo, among others. If you face an error, please feel free to create an issue and I will try to look into it. You are also welcome to fork and push the changes. If you're interested in building on top of our work, we also welcome your contribution.


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